#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 1. Videoyu Yükle
    // Kendi video yolunu buraya yaz
    VideoCapture cap("../data/test_video.mp4"); 

    if (!cap.isOpened()) {
        cerr << "Hata: Video dosyası açılamadı!" << endl;
        return -1;
    }

    Mat prev_frame, curr_frame;
    Mat prev_gray, curr_gray;
    vector<Point2f> prev_pts, curr_pts;

    // İlk kareyi oku
    cap >> prev_frame;
    if (prev_frame.empty()) return 0;

    // Optical flow gri tonlamada (grayscale) çalışır
    cvtColor(prev_frame, prev_gray, COLOR_BGR2GRAY);

    // 2. Takip edilecek özellikleri (feature) bul
    // maxCorners=200, qualityLevel=0.01, minDistance=30
    goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);

    // Görselleştirme için maske
    Mat mask = Mat::zeros(prev_frame.size(), prev_frame.type());

    while (true) {
        cap >> curr_frame;
        if (curr_frame.empty()) break;

        cvtColor(curr_frame, curr_gray, COLOR_BGR2GRAY);

        // 3. Optical Flow Hesapla (Lucas-Kanade)
        vector<uchar> status; // Takip başarılı mı? (1 veya 0)
        vector<float> err;    // Hata oranı
        
        // winSize: Arama penceresi boyutu (21x21 iyidir)
        // maxLevel: Piramit seviyesi (büyük hareketler için 3 iyidir)
        calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err, Size(21, 21), 3);

        // 4. İyi noktaları seç ve çiz
        vector<Point2f> good_new;
        for (uint i = 0; i < prev_pts.size(); i++) {
            // Eğer nokta takibi başarısızsa atla
            if (status[i] == 1) {
                good_new.push_back(curr_pts[i]);

                // Hareket çizgisi çiz (Yeşil)
                line(mask, curr_pts[i], prev_pts[i], Scalar(0, 255, 0), 2);
                // Nokta koy (Kırmızı)
                circle(curr_frame, curr_pts[i], 5, Scalar(0, 0, 255), -1);
            }
        }

        // Sonucu göster
        Mat img;
        add(curr_frame, mask, img);
        imshow("Hafta 1: Optical Flow Takibi", img);

        // 'q' tuşuna basılırsa çık
        if (waitKey(30) == 'q') break;

        // 5. Bir sonraki döngü için 'şu anki' kareyi 'önceki' yap
        prev_gray = curr_gray.clone();
        prev_pts = good_new;

        // Nokta sayısı çok azaldıysa (örn. sahne değiştiyse) yeniden nokta bul
        if (prev_pts.size() < 10) {
            goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);
            mask = Mat::zeros(curr_frame.size(), curr_frame.type()); // Çizgileri temizle
        }
    }

    return 0;
}