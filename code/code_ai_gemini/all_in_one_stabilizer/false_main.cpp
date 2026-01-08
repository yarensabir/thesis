#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

// =============================================================================
// 1. TEMEL VERİ YAPISI VE YARDIMCI FONKSİYONLAR
// =============================================================================

struct TransformParam {
    double dx, dy, da; // x, y, angle

    TransformParam() : dx(0), dy(0), da(0) {}
    TransformParam(double _dx, double _dy, double _da) : dx(_dx), dy(_dy), da(_da) {}

    TransformParam operator+(const TransformParam& p) const {
        return TransformParam(dx + p.dx, dy + p.dy, da + p.da);
    }
    TransformParam operator-(const TransformParam& p) const {
        return TransformParam(dx - p.dx, dy - p.dy, da - p.da);
    }
    TransformParam operator/(double val) const {
        return TransformParam(dx / val, dy / val, da / val);
    }
    void operator+=(const TransformParam& p) {
        dx += p.dx; dy += p.dy; da += p.da;
    }
};

Mat getAffineFromParam(TransformParam p) {
    Mat m = Mat::eye(2, 3, CV_64F);
    m.at<double>(0,0) = cos(p.da);
    m.at<double>(0,1) = -sin(p.da);
    m.at<double>(1,0) = sin(p.da);
    m.at<double>(1,1) = cos(p.da);
    m.at<double>(0,2) = p.dx;
    m.at<double>(1,2) = p.dy;
    return m;
}

// =============================================================================
// 2. ORTAK MOTOR: HAREKET KESTİRİMİ (MOTION ESTIMATOR)
// =============================================================================
class MotionEstimator {
private:
    Mat prev_gray;
    vector<Point2f> prev_pts;
    int max_corners = 50;
    double quality = 0.01;
    double min_dist = 30;

public:
    void initialize(Mat& first_frame) {
        cvtColor(first_frame, prev_gray, COLOR_BGR2GRAY);
        goodFeaturesToTrack(prev_gray, prev_pts, max_corners, quality, min_dist);
    }

    TransformParam estimate(Mat& curr_frame) {
        Mat curr_gray;
        cvtColor(curr_frame, curr_gray, COLOR_BGR2GRAY);

        vector<Point2f> curr_pts;
        vector<uchar> status;
        vector<float> err;
        
        if (prev_pts.size() > 0) {
            calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err, Size(21, 21), 3);
        }

        vector<Point2f> p0, p1;
        for(size_t i=0; i < status.size(); i++) {
            if(status[i]) {
                p0.push_back(prev_pts[i]);
                p1.push_back(curr_pts[i]);
            }
        }

        double dx=0, dy=0, da=0;
        if(p0.size() > 10) {
            Mat T = estimateAffinePartial2D(p0, p1);
            if(!T.empty()) {
                dx = T.at<double>(0,2);
                dy = T.at<double>(1,2);
                da = atan2(T.at<double>(1,0), T.at<double>(0,0));
            }
        }

        prev_gray = curr_gray.clone();
        prev_pts = p1;
        if(prev_pts.size() < 15) { // Nokta sayısı azaldıysa yenile
            goodFeaturesToTrack(prev_gray, prev_pts, max_corners, quality, min_dist);
        }

        return TransformParam(dx, dy, da);
    }
};

// =============================================================================
// 3. FONKSİYON: GERÇEK ZAMANLI STABILIZASYON (PI KAMERA)
// =============================================================================
void runRealTimeStabilization() {
    cout << "[Real-Time Modu] Pi Kamera GStreamer ile aciliyor..." << endl;

    // --- KRİTİK DEĞİŞİKLİK: PI KAMERA İÇİN GSTREAMER PIPELINE ---
    // 640x480 çözünürlük seçtik çünkü işlemciyi yormadan 30 FPS almalıyız.
    // Eğer Legacy Mode kullanıyorsan burayı VideoCapture(0) yapabilirsin.
    string pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink";
    VideoCapture cap(pipeline, CAP_GSTREAMER);

    if(!cap.isOpened()) { 
        cerr << "Hata: Pi Kamera acilamadi! (libcamera calisiyor mu?)" << endl; 
        return; 
    }

    int w = 640;
    int h = 480;
    VideoWriter writer("RealTime_PiCam_Result.mp4", VideoWriter::fourcc('m','p','4','v'), 30, Size(w,h));
    
    // Veri Kaydı
    FILE* fp = fopen("data_realtime.csv", "w");
    fprintf(fp, "frame,raw_x,smooth_x\n");

    MotionEstimator estimator;
    Mat curr;
    
    // Tampon Ayarları
    int BUFFER_SIZE = 30; 
    deque<TransformParam> motion_buffer;
    deque<Mat> frame_buffer;
    TransformParam cumulative_motion(0,0,0);

    cap >> curr;
    if (curr.empty()) { cerr << "Kameradan goruntu gelmedi!" << endl; return; }
    
    estimator.initialize(curr);
    motion_buffer.push_back(TransformParam(0,0,0));
    frame_buffer.push_back(curr.clone());

    cout << "Tampon doluyor (" << BUFFER_SIZE << " kare)... Lutfen bekleyin." << endl;
    int frame_idx = 0;

    // Performans sayacı
    int64 t_start = getTickCount();

    while(true) {
        cap >> curr;
        if(curr.empty()) break;

        // 1. Hareket Kestirimi
        TransformParam motion = estimator.estimate(curr);
        cumulative_motion += motion;

        // 2. Tampona Ekle
        motion_buffer.push_back(cumulative_motion);
        frame_buffer.push_back(curr.clone());

        // 3. İşleme (Buffer Dolduysa)
        if(frame_buffer.size() > BUFFER_SIZE) {
            TransformParam sum(0,0,0);
            for(auto& m : motion_buffer) sum += m;
            TransformParam smoothed_pos = sum / (double)motion_buffer.size();

            Mat target_frame = frame_buffer.front();
            TransformParam target_pos = motion_buffer.front();
            
            // Ters yöne düzeltme (Target - Smooth)
            TransformParam diff = target_pos - smoothed_pos;
            
            Mat M = getAffineFromParam(diff);
            Mat stabilized;
            warpAffine(target_frame, stabilized, M, target_frame.size());

            // Yazma ve Kayıt
            writer.write(stabilized);
            fprintf(fp, "%d, %f, %f\n", frame_idx, target_pos.dx, smoothed_pos.dx);

            // FPS Göstergesi (Opsiyonel: Konsolu kirletmesin diye her 30 karede bir)
            if(frame_idx % 30 == 0) {
                 double fps = getTickFrequency() / (getTickCount() - t_start) * 30;
                 t_start = getTickCount();
                 // cout << "FPS: " << fps << endl; 
            }

            frame_buffer.pop_front();
            motion_buffer.pop_front();
            frame_idx++;
        }
        
        // Çıkış için opsiyonel (Pi'de klavye yoksa çalışmaz, SSH'da Ctrl+C kullan)
        // if(waitKey(1) == 'q') break; 
    }
    
    // Kalanları yaz
    while(!frame_buffer.empty()) {
        writer.write(frame_buffer.front());
        frame_buffer.pop_front();
    }

    fclose(fp);
    cout << "Kayit Tamamlandi: RealTime_PiCam_Result.mp4" << endl;
}

// =============================================================================
// 4. FONKSİYON: ÇEVRİMDIŞI STABILIZASYON (DOSYA İLE)
// =============================================================================
void runOfflineStabilization(string videoPath) {
    cout << "[Offline Modu] Dosya Analiz Ediliyor: " << videoPath << endl;
    
    VideoCapture cap(videoPath);
    if(!cap.isOpened()) { cerr << "Dosya acilamadi!" << endl; return; }

    MotionEstimator estimator;
    Mat curr;
    
    vector<TransformParam> transforms; 
    vector<TransformParam> trajectory;
    TransformParam cumulative(0,0,0);

    cap >> curr;
    estimator.initialize(curr);

    // PASS 1: Analiz
    while(true) {
        cap >> curr;
        if(curr.empty()) break;
        TransformParam motion = estimator.estimate(curr);
        transforms.push_back(motion);
        cumulative += motion;
        trajectory.push_back(cumulative);
    }
    
    cout << "Analiz bitti (" << trajectory.size() << " kare). Yumusatma yapiliyor..." << endl;

    // Yumuşatma (Moving Average)
    vector<TransformParam> smoothed_trajectory;
    int RADIUS = 30;

    for(size_t i=0; i<trajectory.size(); i++) {
        TransformParam sum(0,0,0);
        int count = 0;
        for(int j=-RADIUS; j<=RADIUS; j++) {
            if(i+j >= 0 && i+j < trajectory.size()) {
                sum += trajectory[i+j];
                count++;
            }
        }
        smoothed_trajectory.push_back(sum / count);
    }

    // PASS 2: Video Oluşturma
    cout << "Video yaziliyor..." << endl;
    cap.release();
    cap.open(videoPath);
    
    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    if (fps<=0) fps=30;

    VideoWriter writer("Offline_File_Result.mp4", VideoWriter::fourcc('m','p','4','v'), fps, Size(w,h));
    
    cap >> curr; // İlk kareyi geç

    for(size_t i=0; i<transforms.size(); i++) {
        cap >> curr;
        if(curr.empty()) break;

        TransformParam diff = smoothed_trajectory[i] - trajectory[i];
        
        // Çevrimdışı düzeltme mantığı bazen terstir, görsel olarak kontrol edersin.
        // Genellikle: I_stabilized = Warp(I_original, Smooth - Original)
        Mat M = getAffineFromParam(diff);
        Mat stabilized;
        warpAffine(curr, stabilized, M, curr.size());
        
        writer.write(stabilized);
    }

    cout << "Kayit Tamamlandi: Offline_File_Result.mp4" << endl;
}

// =============================================================================
// 5. MAIN
// =============================================================================
int main() {
    // Çevrimdışı test için kullanılacak video dosyası yolu
    // Lütfen burayı kendi dosya yoluna göre güncelle
    string offlineVideoPath = "/home/pi/Documents/thesis/test_video/ControlCam_7sec.mp4"; 

    int secim;
    cout << "==========================================" << endl;
    cout << "VIDEO STABILIZATION (PI CAMERA OPTIMIZED)" << endl;
    cout << "==========================================" << endl;
    cout << "1. Pi Kamera ile Real-Time Stabilizasyon" << endl;
    cout << "2. Video Dosyasi ile Offline Stabilizasyon" << endl;
    cout << "Secim (1 veya 2): ";
    cin >> secim;

    if (secim == 1) {
        // Parametre almıyor, doğrudan kamerayı açıyor
        runRealTimeStabilization();
    } else if (secim == 2) {
        runOfflineStabilization(offlineVideoPath);
    } else {
        cout << "Gecersiz secim!" << endl;
    }

    return 0;
}