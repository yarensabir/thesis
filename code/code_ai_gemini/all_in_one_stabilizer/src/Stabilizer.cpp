#include "Stabilizer.h"
#include "MotionEstimator.h"
#include "Transform.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>
#include <fstream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// ---------------------------------------------------------
// YARDIMCI SINIF: KALMAN FILTRESI
// ---------------------------------------------------------
struct MotionKalman {
    KalmanFilter KF;
    Mat state, meas;

    MotionKalman() {
        KF.init(6, 3, 0); // 6 Durum (x,y,a, vx,vy,va), 3 Ölçüm (x,y,a)
        setIdentity(KF.transitionMatrix);
        setIdentity(KF.measurementMatrix);
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5)); // Hassasiyet ayarı
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1)); // Gürültü ayarı
        setIdentity(KF.errorCovPost, Scalar::all(1));
        meas = Mat::zeros(3, 1, CV_32F);
    }

    TransformParam update(TransformParam raw) {
        KF.predict();
        meas.at<float>(0) = (float)raw.dx;
        meas.at<float>(1) = (float)raw.dy;
        meas.at<float>(2) = (float)raw.da;
        Mat estimated = KF.correct(meas);
        return TransformParam(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
    }
};

// ---------------------------------------------------------
// 1. GERÇEK ZAMANLI FONKSİYON
// ---------------------------------------------------------
void runRealTimeStabilization(RealTimeMethod method) {
    cout << "[Real-Time] Pi Kamera GStreamer ile aciliyor..." << endl;
    if (method == RT_KALMAN_FILTER) cout << ">> Yontem: KALMAN FILTRESI" << endl;
    else cout << ">> Yontem: KAYAN PENCERE (BUFFER)" << endl;

    string pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink";
    VideoCapture cap(pipeline, CAP_GSTREAMER);

    if(!cap.isOpened()) { cerr << "Hata: Kamera acilamadi!" << endl; return; }

    VideoWriter writer("RealTime_Result.mp4", VideoWriter::fourcc('m','p','4','v'), 30, Size(640,480));
    FILE* fp = fopen("data_realtime.csv", "w");
    fprintf(fp, "frame,raw_x,smooth_x\n");

    MotionEstimator estimator;
    MotionKalman kf; // Kalman nesnesi
    Mat curr;
    
    int BUFFER_SIZE = 30; 
    deque<TransformParam> motion_buffer;
    deque<Mat> frame_buffer;
    TransformParam cumulative_motion(0,0,0);
    TransformParam smoothed_pos(0,0,0); // Kalman için anlık tutucu

    cap >> curr;
    if (curr.empty()) return;
    estimator.initialize(curr);

    // Başlangıç değerleri
    motion_buffer.push_back(TransformParam(0,0,0));
    frame_buffer.push_back(curr.clone());

    int frame_idx = 0;
    cout << "Islem basliyor..." << endl;

    while(true) {
        cap >> curr;
        if(curr.empty()) break;

        // A. HAREKETİ BUL
        TransformParam motion = estimator.estimate(curr);
        cumulative_motion += motion;

        // B. YÖNTEM SEÇİMİ (BURASI KRİTİK NOKTA)
        if (method == RT_KALMAN_FILTER) {
            // Kalman anlık çalışır, buffer beklemesine gerek yoktur ama
            // görüntüyle senkronize gitmek için buffer yapısını koruyoruz.
            // Sadece smoothed_pos hesabını değiştiriyoruz.
            smoothed_pos = kf.update(cumulative_motion);
        }
        
        // Verileri Buffer'a at
        motion_buffer.push_back(cumulative_motion);
        frame_buffer.push_back(curr.clone());

        // C. Buffer Dolduysa İşle
        if(frame_buffer.size() > BUFFER_SIZE) {
            
            // Eğer Yöntem BUFFER ise ortalamayı burada alıyoruz
            if (method == RT_SLIDING_WINDOW) {
                TransformParam sum(0,0,0);
                for(const auto& m : motion_buffer) sum += m;
                smoothed_pos = sum / (double)motion_buffer.size();
            }

            // Kalman zaten yukarıda hesaplanmıştı, smoothed_pos hazır.

            Mat target_frame = frame_buffer.front();
            TransformParam target_pos = motion_buffer.front(); // Orijinal konum
            
            // Fark: Hedef - Pürüzsüz
            TransformParam diff = target_pos - smoothed_pos;
            
            Mat M = getAffineFromParam(diff);
            Mat stabilized;
            warpAffine(target_frame, stabilized, M, target_frame.size());

            writer.write(stabilized);
            fprintf(fp, "%d, %f, %f\n", frame_idx, target_pos.dx, smoothed_pos.dx);

            frame_buffer.pop_front();
            motion_buffer.pop_front();
            frame_idx++;
        }
    }
    
    // Kalanları boşalt
    while(!frame_buffer.empty()) {
        writer.write(frame_buffer.front());
        frame_buffer.pop_front();
    }
    fclose(fp);
    cout << "RealTime Bitti." << endl;
}

// ---------------------------------------------------------
// 2. ÇEVRİMDIŞI FONKSİYON
// ---------------------------------------------------------
void runOfflineStabilization(string videoPath, OfflineMethod method) {
    cout << "[Offline] Analiz yapiliyor..." << endl;
    if (method == OFF_GAUSSIAN) cout << ">> Yontem: GAUSSIAN SMOOTHING" << endl;
    else cout << ">> Yontem: MOVING AVERAGE" << endl;

    VideoCapture cap(videoPath);
    if(!cap.isOpened()) { cerr << "Dosya yok!" << endl; return; }

    MotionEstimator estimator;
    Mat curr;
    vector<TransformParam> trajectory; // Kümülatif yörünge
    vector<TransformParam> transforms; // Anlık hareketler
    TransformParam cumulative(0,0,0);

    cap >> curr;
    estimator.initialize(curr);

    // PASS 1: Veri Topla
    while(true) {
        cap >> curr;
        if(curr.empty()) break;
        TransformParam motion = estimator.estimate(curr);
        transforms.push_back(motion);
        cumulative += motion;
        trajectory.push_back(cumulative);
    }
    
    // PASS 2: Yumuşatma (YÖNTEM SEÇİMİ BURADA)
    cout << "Yumusatma uygulaniyor..." << endl;
    vector<TransformParam> smoothed_trajectory;
    int RADIUS = 30; // Pencere yarıçapı

    for(size_t i=0; i<trajectory.size(); i++) {
        TransformParam sum(0,0,0);
        double total_weight = 0;

        for(int j=-RADIUS; j<=RADIUS; j++) {
            if(i+j >= 0 && i+j < trajectory.size()) {
                
                if (method == OFF_MOVING_AVERAGE) {
                    // YÖNTEM A: Eşit Ağırlık (1.0)
                    sum += trajectory[i+j];
                    total_weight += 1.0;
                } 
                else if (method == OFF_GAUSSIAN) {
                    // YÖNTEM B: Gaussian Ağırlık
                    // Sigma = Radius / 3 standart bir kuraldır
                    double sigma = RADIUS / 3.0;
                    double weight = exp(-(j*j) / (2 * sigma * sigma));
                    
                    // TransformParam ile double çarpımı operatörü tanımlamadığımız için elle yapıyoruz:
                    TransformParam p = trajectory[i+j];
                    sum.dx += p.dx * weight;
                    sum.dy += p.dy * weight;
                    sum.da += p.da * weight;
                    
                    total_weight += weight;
                }
            }
        }
        
        // Ağırlıklı Ortalama
        if (method == OFF_MOVING_AVERAGE) {
            smoothed_trajectory.push_back(sum / total_weight);
        } else {
            // Gaussian için elle bölme
            smoothed_trajectory.push_back(TransformParam(sum.dx/total_weight, sum.dy/total_weight, sum.da/total_weight));
        }
    }

    // PASS 3: Video Yazma
    cout << "Video olusturuluyor..." << endl;
    cap.release(); cap.open(videoPath);
    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter writer("Offline_Result.mp4", VideoWriter::fourcc('m','p','4','v'), 30, Size(w,h));
    
    cap >> curr; 
    for(size_t i=0; i<transforms.size(); i++) {
        cap >> curr;
        if(curr.empty()) break;
        TransformParam diff = smoothed_trajectory[i] - trajectory[i];
        Mat M = getAffineFromParam(diff);
        Mat stabilized;
        warpAffine(curr, stabilized, M, curr.size());
        writer.write(stabilized);
    }
    cout << "Offline Bitti." << endl;
}