#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

// --- Sabitler ve Ayarlar ---
const string VIDEO_PATH = "../s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820.mp4";
const string OUTPUT_PATH = "stabilized_final_orb.avi"; 
const int SMOOTHING_RADIUS = 30;         // Hareket yumuşatma için pencere boyutu
const int INTERPOLATION_METHOD = INTER_LINEAR; // Yeniden örnekleme yöntemi (Bilineer)
const int ORB_FEATURES = 500;            // Tespit edilecek maksimum ORB özniteliği sayısı

// --- Veri Yapıları ---
struct TransformParam {
    double dx; // Öteleme X
    double dy; // Öteleme Y
    double da; // Rotasyon Açısı
};

struct Trajectory {
    double x;
    double y;
    double a;
};

// --- Yardımcı Fonksiyonlar ---

/**
 * Kayan ortalama (Moving Average) filtresi ile yörüngeyi yumuşatır.
 * (Hareket Telafisi ve Düzeltme aşamasının çekirdeği)
 */
void smooth(const vector<double>& trajectory, vector<double>& smoothed_trajectory) {
    smoothed_trajectory.clear();
    for (size_t i = 0; i < trajectory.size(); ++i) {
        double sum = 0;
        int count = 0;
        int low = max(0, (int)i - SMOOTHING_RADIUS);
        int high = min((int)i + SMOOTHING_RADIUS, (int)trajectory.size() - 1);

        for (int j = low; j <= high; ++j) {
            sum += trajectory[j];
            count++;
        }
        smoothed_trajectory.push_back(sum / count);
    }
}

template<typename T, typename F>
vector<double> extract_components(const vector<T>& data, F extractor) {
    vector<double> components;
    for (const auto& item : data) {
        components.push_back(extractor(item));
    }
    return components;
}

// --------------------------------------------------

int main() {
    // --- 1. Başlangıç ve Giriş/Çıkış Ayarları ---
    VideoCapture cap(VIDEO_PATH);

    if (!cap.isOpened()) {
        cerr << "HATA: Video dosyası açılamadı: " << VIDEO_PATH << endl;
        return -1;
    }

    int n_frames = (int)cap.get(CAP_PROP_FRAME_COUNT);
    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);

    VideoWriter output_cap(OUTPUT_PATH, 
                           VideoWriter::fourcc('X','V','I','D'), 
                           fps, Size(w, h));

    if (!output_cap.isOpened()) {
        cerr << "HATA: Çıktı video dosyası (" << OUTPUT_PATH << ") oluşturulamadı." << endl;
        return -1;
    }
    
    cout << "Video Stabilizasyonu Basliyor. Toplam Kare: " << n_frames << ", FPS: " << fps << endl;

    // Öznitelik Çıkarıcı ve Eşleştirici
    Ptr<ORB> detector = ORB::create(ORB_FEATURES);
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);

    Mat prev, prev_gray, prev_desc;
    vector<KeyPoint> prev_kp;

    cap >> prev;
    if (prev.empty()) return -1;
    cvtColor(prev, prev_gray, COLOR_BGR2GRAY);
    detector->detectAndCompute(prev_gray, noArray(), prev_kp, prev_desc);

    vector<TransformParam> transforms;
    
    // --- 2. Aşama: Hareket Tahmini ve Tespiti (Öznitelik Eşleştirme) ---
    for (int i = 1; i < n_frames; ++i) {
        Mat curr, curr_gray, curr_desc;
        vector<KeyPoint> curr_kp;
        cap >> curr;
        if (curr.empty()) break;
        cvtColor(curr, curr_gray, COLOR_BGR2GRAY);
        
        // Öznitelikleri Çıkar ve Eşleştir
        detector->detectAndCompute(curr_gray, noArray(), curr_kp, curr_desc);

        vector<DMatch> matches;
        if (!prev_desc.empty() && !curr_desc.empty()) {
            matcher->match(prev_desc, curr_desc, matches);
        }

        // İyi eşleşmeleri filtrele
        double min_dist = 1000.0;
        for (const auto& match : matches) {
            if (match.distance < min_dist) min_dist = match.distance;
        }

        vector<Point2f> prev_pts, curr_pts;
        for (const auto& match : matches) {
            if (match.distance < 3 * min_dist) { 
                prev_pts.push_back(prev_kp[match.queryIdx].pt);
                curr_pts.push_back(curr_kp[match.trainIdx].pt);
            }
        }
        
        // Rijit Dönüşüm Matrisi Tahmini (RANSAC ile)
        Mat T = Mat::eye(2, 3, CV_64F);
        if (prev_pts.size() > 5) {
            T = estimateAffinePartial2D(prev_pts, curr_pts);
        }
        
        // Dönüşüm parametrelerini çıkar
        double dx = T.at<double>(0, 2);
        double dy = T.at<double>(1, 2);
        double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0)); 

        transforms.push_back({dx, dy, da});

        // Bir sonraki döngü için güncelle
        prev_gray = curr_gray.clone();
        prev_desc = curr_desc.clone();
        prev_kp = curr_kp;

        if (i % 100 == 0 || i == n_frames - 1) {
            cout << "Hareket Tahmini: " << (int)((double)i / n_frames * 100) << "%" << "\r" << flush;
        }
    }
    cout << "\nHareket Tahmini Tamamlandi." << endl;

    // --- 3. Aşama: Hareket Telafisi ve Düzeltme ---
    vector<Trajectory> trajectory;
    double a = 0; double x = 0; double y = 0;

    for (const auto& trans : transforms) {
        x += trans.dx;
        y += trans.dy;
        a += trans.da;
        trajectory.push_back({x, y, a});
    }

    // Yumuşatılmış yörüngeyi hesapla
    vector<double> smoothed_x, smoothed_y, smoothed_a;
    smooth(extract_components(trajectory, [](const Trajectory& t){ return t.x; }), smoothed_x);
    smooth(extract_components(trajectory, [](const Trajectory& t){ return t.y; }), smoothed_y);
    smooth(extract_components(trajectory, [](const Trajectory& t){ return t.a; }), smoothed_a);
    
    // İlk kareyi doğrudan çıktıya yaz
    cap.set(CAP_PROP_POS_FRAMES, 0);
    cap >> prev;
    output_cap << prev; 

    // --- 4. Aşama: Görüntü Yeniden Yapılandırma (Resampling/Interpolation) ---
    for (int i = 0; i < n_frames - 1; ++i) {
        Mat curr, curr_stabilized;
        cap >> curr;
        if (curr.empty()) break;

        // Düzeltme hareketini hesapla
        double diff_x = smoothed_x[i] - trajectory[i].x;
        double diff_y = smoothed_y[i] - trajectory[i].y;
        double diff_a = smoothed_a[i] - trajectory[i].a;

        double dx_stabilized = transforms[i].dx + diff_x;
        double dy_stabilized = transforms[i].dy + diff_y;
        double da_stabilized = transforms[i].da + diff_a;

        // Düzeltilmiş Dönüşüm Matrisini oluştur
        Mat T_stabilized = Mat::zeros(2, 3, CV_64F);
        T_stabilized.at<double>(0, 0) = cos(da_stabilized);
        T_stabilized.at<double>(0, 1) = -sin(da_stabilized);
        T_stabilized.at<double>(1, 0) = sin(da_stabilized);
        T_stabilized.at<double>(1, 1) = cos(da_stabilized);
        T_stabilized.at<double>(0, 2) = dx_stabilized;
        T_stabilized.at<double>(1, 2) = dy_stabilized;

        // Yeniden Örnekleme (Enterpolasyon) Adımı:
        warpAffine(curr, curr_stabilized, 
                   T_stabilized, 
                   curr.size(), 
                   INTERPOLATION_METHOD,  // Bilineer
                   BORDER_REFLECT_101);   // Sınır doldurma

        output_cap << curr_stabilized;
        
        if (i % 100 == 0 || i == n_frames - 2) {
            cout << "Stabilizasyon Uygulaniyor: " << (int)((double)(i + 1) / (n_frames - 1) * 100) << "%" << "\r" << flush;
        }
    }

    cap.release();
    output_cap.release();

    cout << "\nStabilizasyon başarıyla tamamlandı." << endl;
    cout << "Çıktı dosyasi: " << OUTPUT_PATH << endl;

    return 0;
}