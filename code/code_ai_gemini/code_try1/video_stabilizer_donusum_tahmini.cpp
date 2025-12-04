#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

// Basit 2x3 dönüşüm matrisi yapısı
struct TransformParam {
    double dx;
    double dy;
    double da; // açı
};

// Yörünge noktası yapısı
struct Trajectory {
    double x;
    double y;
    double a;
};

// Yumuşatma penceresi yarıçapı
const int SMOOTHING_RADIUS = 30; 

// Kayan ortalama (Moving Average) ile yumuşatma fonksiyonu
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

// lambda kullanımı için yardımcı fonksiyon
template<typename T, typename F>
vector<double> extract_components(const vector<T>& data, F extractor) {
    vector<double> components;
    for (const auto& item : data) {
        components.push_back(extractor(item));
    }
    return components;
}

int main() {
    // --- 1. Adım: Giriş ve Çıkış Dosya Yollarını Tanımlama ---
    // Kodunuz code_ai klasöründe olduğundan, video yoluna göreceli erişim:
    const string VIDEO_PATH = "../s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820.mp4";
    
    // Çıktı codec'ini XVID olarak değiştirdim ve uzantıyı .avi yaptım.
    const string OUTPUT_PATH = "stabilized_output.avi"; 

    VideoCapture cap(VIDEO_PATH);

    if (!cap.isOpened()) {
        cerr << "HATA: Video dosyası açılamadı: " << VIDEO_PATH << endl;
        return -1;
    }

    int n_frames = (int)cap.get(CAP_PROP_FRAME_COUNT);
    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);

    // Çıktı videosunu kaydetmek için VideoWriter
    VideoWriter output_cap(OUTPUT_PATH, 
                           VideoWriter::fourcc('X','V','I','D'), // Daha uyumlu bir codec
                           fps, Size(w, h));

    if (!output_cap.isOpened()) {
        cerr << "HATA: Çıktı video dosyası (" << OUTPUT_PATH << ") oluşturulamadı." << endl;
        return -1;
    }
    
    cout << "Video Stabilizasyonu Basliyor. Toplam Kare: " << n_frames << ", FPS: " << fps << endl;

    Mat prev, prev_gray;
    cap >> prev;
    if (prev.empty()) {
        cerr << "Video bos." << endl;
        return -1;
    }
    cvtColor(prev, prev_gray, COLOR_BGR2GRAY);

    vector<TransformParam> transforms;
    
    // --- 2. Adım: Ardışık Çerçeveler Arasındaki Dönüşümü Tahmin Etme ---
    for (int i = 1; i < n_frames; ++i) {
        Mat curr, curr_gray;
        cap >> curr;
        if (curr.empty()) break;
        cvtColor(curr, curr_gray, COLOR_BGR2GRAY);
        
        // Optik Akış (Optical Flow) ve Rijit Dönüşüm Tahmini
        vector<Point2f> prev_pts, curr_pts;
        goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);
        
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);

        vector<Point2f> prev_pts2, curr_pts2;
        for (size_t k = 0; k < status.size(); k++) {
            if (status[k]) {
                prev_pts2.push_back(prev_pts[k]);
                curr_pts2.push_back(curr_pts[k]);
            }
        }

        Mat T;
        if (prev_pts2.size() > 5) {
            T = estimateAffinePartial2D(prev_pts2, curr_pts2);
        } else {
            T = Mat::eye(2, 3, CV_64F);
        }

        double dx = T.at<double>(0, 2);
        double dy = T.at<double>(1, 2);
        double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0)); 

        transforms.push_back({dx, dy, da});

        prev_gray = curr_gray.clone();
        
        // İlerleme çubuğu (İsteğe bağlı)
        if (i % 100 == 0 || i == n_frames - 1) {
            cout << "Hareket Tahmini: " << (int)((double)i / n_frames * 100) << "%" << "\r" << flush;
        }
    }
    cout << "Hareket Tahmini Tamamlandi. %100" << endl;


    // --- 3. Adım: Yörüngeyi Hesaplama ve Yumuşatma ---
    vector<Trajectory> trajectory;
    double a = 0;
    double x = 0;
    double y = 0;

    for (const auto& trans : transforms) {
        x += trans.dx;
        y += trans.dy;
        a += trans.da;
        trajectory.push_back({x, y, a});
    }

    // Yumuşatılmış Yörüngeyi hesapla
    vector<double> traj_x = extract_components(trajectory, [](const Trajectory& t){ return t.x; });
    vector<double> traj_y = extract_components(trajectory, [](const Trajectory& t){ return t.y; });
    vector<double> traj_a = extract_components(trajectory, [](const Trajectory& t){ return t.a; });
    
    vector<double> smoothed_x, smoothed_y, smoothed_a;
    smooth(traj_x, smoothed_x);
    smooth(traj_y, smoothed_y);
    smooth(traj_a, smoothed_a);


    // --- 4. Adım: Düzeltme Dönüşümlerini Uygulama ve Kaydetme ---
    
    // Videoyu tekrar başa sar ve ilk kareyi doğrudan yaz
    cap.set(CAP_PROP_POS_FRAMES, 0);
    cap >> prev;
    output_cap << prev; 

    for (int i = 0; i < n_frames - 1; ++i) {
        Mat curr, curr_stabilized;
        cap >> curr;
        if (curr.empty()) break;

        // Düzeltme Matrisi ve Stabilize Edilmiş Hareket hesaplamaları
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

        // Çerçeveye düzeltilmiş dönüşümü uygula
        warpAffine(curr, curr_stabilized, T_stabilized, curr.size(), INTER_LINEAR, BORDER_REFLECT_101);

        // **SADECE KAYDETME İŞLEMİ YAPILIYOR**
        output_cap << curr_stabilized;
        prev = curr.clone(); 
        
        // İlerleme çubuğu
        if (i % 100 == 0 || i == n_frames - 2) {
            cout << "Stabilizasyon Uygulanmasi: " << (int)((double)(i + 1) / (n_frames - 1) * 100) << "%" << "\r" << flush;
        }
    }

    cap.release();
    output_cap.release();
    // destroyAllWindows() çağrılmasına gerek yok, çünkü pencere açılmadı.

    cout << "\nStabilizasyon tamamlandı ve sadece kaydedildi." << endl;
    cout << "Çıktı dosyasi: " << OUTPUT_PATH << " (thesis/code_ai klasorunde)" << endl;

    return 0;
}