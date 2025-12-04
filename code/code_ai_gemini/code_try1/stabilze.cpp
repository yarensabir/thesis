#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace cv;

// Sabitler
const int MAX_CORNERS = 200; // Takip edilecek maksimum köşe sayısı
const double FEATURE_QUALITY = 0.01; // Köşe kalitesi
const double MIN_DISTANCE = 30.0; // İki köşe arasındaki minimum mesafe (piksellerle)
const int SMOOTHING_RADIUS = 30; // Kare sayısında yumuşatma yarıçapı (Daha büyük = daha kararlı, daha az tepkisel)
const int HORIZONTAL_BORDER_CROP = 20; // Stabilizasyondan kaynaklanan siyah kenarları azaltmak için yatay kırpma (piksellerle)
const bool SHOW_DISPLAY = false; // <<< YENI: Grafik arayuzu (imshow) gostermek icin bayrak. Hata alindigi icin 'false' olarak ayarlandi.

// --- Kullanıcının istediği sabit video yolları ---
// Not: Bu yollar, programın 'thesis/code_ai' klasöründe çalıştırıldığı varsayılarak ayarlanmıştır.
const string INPUT_VIDEO_PATH = "../s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820.mp4";
// Çıktı dosyası, giriş videosunun bulunduğu klasöre kaydedilecektir.

/**
 * @brief Dönüşüm parametrelerini (x, y ve açı) tutan yapı.
 */
struct TransformParam
{
    TransformParam() : dx(0), dy(0), da(0) {}
    TransformParam(double _dx, double _dy, double _da) : dx(_dx), dy(_dy), da(_da) {}
    
    double dx; // X yönündeki yer değiştirme
    double dy; // Y yönündeki yer değiştirme
    double da; // Açı (radyan)
};

/**
 * @brief Kümülatif yörüngeyi (trajectory) tutan yapı.
 */
struct Trajectory
{
    Trajectory() : x(0), y(0), a(0) {}
    Trajectory(double _x, double _y, double _a) : x(_x), y(_y), a(_a) {}
    
    double x; // Kümülatif X
    double y; // Kümülatif Y
    double a; // Kümülatif Açı
    
    // Toplama operatörü
    friend Trajectory operator+(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x + c2.x, c1.y + c2.y, c1.a + c2.a);
    }
    // Çıkarma operatörü
    friend Trajectory operator-(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x - c2.x, c1.y - c2.y, c1.a - c2.a);
    }
};

/**
 * @brief Görüntü dönüşüm matrisini uygulayıp stabilizasyon sınırlarını hesaplar.
 * @param frame_in: Giriş karesi
 * @param T: Dönüşüm matrisi (2x3)
 * @param border_crop: Kırpma miktarı
 * @return Dönüştürülmüş ve kırpılmış kare
 */
Mat applyTransform(Mat &frame_in, Mat &T, int border_crop)
{
    Mat frame_out;
    // Afin dönüşümü uygula
    warpAffine(frame_in, frame_out, T, frame_in.size());
    
    // Siyah kenarları kırp
    int vert_border = border_crop * frame_in.rows / frame_in.cols;
    
    Rect roi(border_crop, vert_border, 
             frame_out.cols - 2 * border_crop, 
             frame_out.rows - 2 * vert_border);
             
    if (roi.width > 0 && roi.height > 0) {
        frame_out = frame_out(roi);
        // Orijinal boyutuna geri ölçeklendir (daha iyi karşılaştırma için)
        resize(frame_out, frame_out, frame_in.size());
    }
    return frame_out;
}

int main(int argc, char **argv)
{
    // Komut satırı argümanı kontrolü kaldırıldı, yol sabitlendi.
    
    // 1. Video yakalama nesnesini aç
    VideoCapture cap(INPUT_VIDEO_PATH);
    if (!cap.isOpened()) {
        cerr << "Hata: Video dosyasi acilamadi: " << INPUT_VIDEO_PATH << endl;
        cerr << "Lutfen dosya yolunu ve izinleri kontrol edin." << endl;
        return -1;
    }

    Mat prev, cur, prev_grey, cur_grey;
    cap >> prev; // İlk kareyi al
    if (prev.empty()) {
        cerr << "Hata: Video dosyasi bos veya ilk kare okunamadi." << endl;
        return -1;
    }
    cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

    // Tüm kareler arası dönüşümleri saklamak için
    vector<TransformParam> prev_to_cur_transform;
    prev_to_cur_transform.reserve(cap.get(CAP_PROP_FRAME_COUNT));

    Mat T; // Mevcut Dönüşüm Matrisi
    Mat last_T = Mat::eye(2, 3, CV_64F); // Son başarılı dönüşümü tutar (Başlangıçta kimlik matrisi)
    
    int k = 1;
    long max_frames = cap.get(CAP_PROP_FRAME_COUNT);

    cout << "Video Stabilizasyonu Baslatiliyor... Toplam kare: " << max_frames << endl;
    cout << "Giris Dosyasi: " << INPUT_VIDEO_PATH << endl;

    while (true) {
        // İkinci kareyi al
        cap >> cur;
        if (cur.empty()) break;
        
        cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

        // --- 1. Hareket Tahmini (Feature Tracking) ---
        
        vector<Point2f> prev_pts, cur_pts;
        vector<uchar> status;
        vector<float> err;

        // Köşe noktalarını tespit et
        goodFeaturesToTrack(prev_grey, prev_pts, MAX_CORNERS, FEATURE_QUALITY, MIN_DISTANCE);
        
        // Optik akışı hesapla (noktaların hareketini)
        calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_pts, cur_pts, status, err);

        // Kötü eşleşmeleri temizle
        vector<Point2f> prev_good_pts, cur_good_pts;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                prev_good_pts.push_back(prev_pts[i]);
                cur_good_pts.push_back(cur_pts[i]);
            }
        }
        
        // Afin dönüşümü tahmin et (yer değiştirme + döndürme).
        // RANSAC, aykırı değerleri otomatik olarak temizler. 3.0 piksel eşiği kullanıldı.
        T = estimateAffine2D(prev_good_pts, cur_good_pts, noArray(), RANSAC, 3.0); 

        // Nadir durumlarda dönüşüm bulunamazsa, son geçerli dönüşümü kullan
        if (T.empty()) {
            T = last_T; // Son başarılı T matrisini kullan
            cerr << "Uyari: Kare " << k << " icin donusum bulunamadi, son T kullaniliyor." << endl;
        } else {
            // Başarılı T'yi bir sonraki döngü için sakla
            T.copyTo(last_T);
        }

        // Dönüşüm matrisini (T) bileşenlerine ayır
        double dx = T.at<double>(0, 2);
        double dy = T.at<double>(1, 2);
        double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));
        
        prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

        // Bir sonraki döngü için hazırlık
        cur.copyTo(prev);
        cur_grey.copyTo(prev_grey);
        
        cout << "Kare: " << k << "/" << max_frames << " - Iyi Akis Noktalari: " << prev_good_pts.size() << endl;
        k++;
    }

    // Tüm kareler okunduktan sonra video başa sarılır
    cap.set(CAP_PROP_POS_FRAMES, 0);

    // --- 2. Yörüngeyi Hesapla ---
    
    double a = 0;
    double x = 0;
    double y = 0;
    vector<Trajectory> trajectory;
    for (size_t i = 0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;
        trajectory.push_back(Trajectory(x, y, a));
    }

    // --- 3. Yörüngeyi Yumuşat ---
    
    vector<Trajectory> smoothed_trajectory;
    // Yörüngeyi yumuşatma penceresi çok büyükse uyarı ver
    if (SMOOTHING_RADIUS * 2 + 1 >= trajectory.size()) {
         cerr << "Uyari: Yumusatma yaricapi (" << SMOOTHING_RADIUS << ") video uzunluguna gore cok buyuk olabilir. Tum video yumusatiliyor." << endl;
    }

    for (size_t i = 0; i < trajectory.size(); i++) {
        double sum_x = 0;
        double sum_y = 0;
        double sum_a = 0;
        int count = 0;
        
        // Kayan ortalama penceresi (Sliding Average Window)
        int low = max(0, (int)i - SMOOTHING_RADIUS);
        int high = min((int)i + SMOOTHING_RADIUS, (int)trajectory.size() - 1);

        for (int j = low; j <= high; j++) {
            sum_x += trajectory[j].x;
            sum_y += trajectory[j].y;
            sum_a += trajectory[j].a;
            count++;
        }
        
        smoothed_trajectory.push_back(Trajectory(sum_x / count, sum_y / count, sum_a / count));
    }

    // --- 4. Yeni Dönüşümleri Hesapla ve 5. Uygula ---
    
    VideoWriter outputVideo;
    Mat canvas; // Karşılaştırma için kullanılacak tuval
    
    // Output dosyası adı hazırlığı (Giriş dosya adından türet)
    // Sadece dosya adını alıp çıktı yolunu buna göre ayarlıyoruz.
    string inputFilename = INPUT_VIDEO_PATH;
    size_t lastSlash = inputFilename.find_last_of("/\\");
    string videoDir = (lastSlash == string::npos) ? "" : inputFilename.substr(0, lastSlash + 1);
    string videoBaseName = (lastSlash == string::npos) ? inputFilename : inputFilename.substr(lastSlash + 1);
    
    size_t lastDot = videoBaseName.find_last_of('.');
    string baseFilename = (lastDot == string::npos) ? videoBaseName : videoBaseName.substr(0, lastDot);
    
    // Çıktı dosyası, giriş videosunun klasöründe kaydedilir.
    string outputFilename = videoDir + baseFilename + "_stabilized.avi";
    
    // Video ayarlarını al
    int fourcc = VideoWriter::fourcc('M','J','P','G'); // Daha yaygın bir codec kullan
    double fps = cap.get(CAP_PROP_FPS);
    Size size((int)cap.get(CAP_PROP_FRAME_WIDTH) * 2 + 10, (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    
    outputVideo.open(outputFilename, fourcc, fps, size, true);

    if (!outputVideo.isOpened()) {
        cerr << "Hata: Cikti video dosyasi olusturulamadi: " << outputFilename << endl;
        cerr << "Lutfen cikti klasorune yazma izinlerini kontrol edin." << endl;
        return -1;
    }

    k = 1; // Kare sayacını sıfırla
    Mat cur_stabilized;
    for (int i = 0; i < prev_to_cur_transform.size(); i++) {
        
        // Yeni kareyi al
        cap >> cur;
        if (cur.empty()) break;
        
        // Gerçek kümülatif yörünge
        Trajectory actual_trajectory = trajectory[i];
        // Yumuşatılmış hedef yörünge
        Trajectory target_trajectory = smoothed_trajectory[i];

        // Dengeleyici hareket (Hedef - Gerçek)
        double diff_x = target_trajectory.x - actual_trajectory.x;
        double diff_y = target_trajectory.y - actual_trajectory.y;
        double diff_a = target_trajectory.a - actual_trajectory.a;

        // Kareler arası dönüşüme dengeleme hareketini ekle
        double new_dx = prev_to_cur_transform[i].dx + diff_x;
        double new_dy = prev_to_cur_transform[i].dy + diff_y;
        double new_da = prev_to_cur_transform[i].da + diff_a;
        
        // Yeni dönüşüm matrisini oluştur
        T = Mat::eye(2, 3, CV_64F);
        T.at<double>(0, 0) = cos(new_da);
        T.at<double>(0, 1) = -sin(new_da);
        T.at<double>(1, 0) = sin(new_da);
        T.at<double>(1, 1) = cos(new_da);
        T.at<double>(0, 2) = new_dx;
        T.at<double>(1, 2) = new_dy;
        
        // Dönüşümü kareye uygula
        cur_stabilized = applyTransform(cur, T, HORIZONTAL_BORDER_CROP);

        // Orijinal ve stabilize edilmiş kareleri yan yana göster
        canvas = Mat::zeros(cur.rows, cur.cols * 2 + 10, cur.type());
        cur.copyTo(canvas(Range::all(), Range(0, cur.cols)));
        cur_stabilized.copyTo(canvas(Range::all(), Range(cur.cols + 10, cur.cols * 2 + 10)));
        
        // Orijinal ve stabilize edilmiş kareler başlıklarını ekle
        putText(canvas, "Orijinal", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
        putText(canvas, "Stabilize Edilmis", Point(cur.cols + 30, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);

        outputVideo << canvas;
        
        if (SHOW_DISPLAY) { // Sadece bayrak true ise arayüzü göster
            imshow("Stabilizasyon (Orijinal vs Stabilize Edilmis)", canvas);
            waitKey(1);
        }

        k++;
    }
    
    cout << "Stabilizasyon tamamlandi. Cikti dosyasi: " << outputFilename << endl;

    cap.release();
    outputVideo.release();
    
    if (SHOW_DISPLAY) {
        destroyAllWindows();
    }
    return 0;
}