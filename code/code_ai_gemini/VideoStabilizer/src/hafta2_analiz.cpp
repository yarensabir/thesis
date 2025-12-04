#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

/**
 * [YARDIMCI FONKSIYON]
 * Ekrana görüntü basar ve çıkış tuşunu kontrol eder.
 * Geri dönüş değeri: Eğer 'q' tuşuna basılırsa true döner (çıkış sinyali).
 */
bool displayFrame(const Mat& img) {
    imshow("Optical Flow Takibi", img);
    // 30ms bekle, eğer 'q'ya basılırsa true döndür
    if (waitKey(30) == 'q') {
        return true; 
    }
    return false;
}

int main() {
    
    FILE* fp = fopen("/home/pi/Documents/thesis/code/code_ai_gemini/VideoStabilizer/motion_data_full.csv", "w");
    fprintf(fp, "frame,dx,dy,angle\n"); // Başlık satırı
    int frame_idx = 0; // Kare indeksini 0'dan başlatıyoruz
    
    
    // 1. Videoyu Yükle
    // Kendi video yolunu buraya yaz
    VideoCapture cap("/home/pi/Documents/thesis/test_video/ControlCam_20200930_104820.mp4"); 
    //VideoCapture cap("/home/pi/Documents/thesis/test_video/ControlCam_7sec.mp4"); 

    if (!cap.isOpened()) {
        cerr << "Hata: Video dosyası açılamadı!" << endl;
        return -1;
    }

    // -------------------------------------------------------------
    //  Video Kayıt Ayarları (VideoWriter Başlatma)
    // -------------------------------------------------------------
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);

    // Çıktı dosyası adı: output.mp4
    // Codec: 'm', 'p', '4', 'v' (Linux/Windows uyumlu MP4)
    VideoWriter writer("/home/pi/Documents/thesis/output_videos/code_ai_gemini/video/OpticalFlow_full.mp4", 
                       VideoWriter::fourcc('m', 'p', '4', 'v'), 
                       fps, 
                       Size(frame_width, frame_height));

    if (!writer.isOpened()) {
        cerr << "Hata: Video yazıcısı başlatılamadı!" << endl;
        return -1;
    }
    // -------------------------------------------------------------

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
    //goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30); // köşe bul
    //200 nokta raspberry pi'yi zorluyor olabilir, düşürüyoruz
    goodFeaturesToTrack(prev_gray, prev_pts, 50, 0.01, 30);

    // Görselleştirme için maske
    Mat mask = Mat::zeros(prev_frame.size(), prev_frame.type());

    while (true) {
        cap >> curr_frame;
        if (curr_frame.empty()) break;
        
        frame_idx++; // Her yeni karede sayacı artır

        cvtColor(curr_frame, curr_gray, COLOR_BGR2GRAY);

        // 3. Optical Flow Hesapla (Lucas-Kanade)
        vector<uchar> status; // Takip başarılı mı? (1 veya 0)
        vector<float> err;    // Hata oranı
        
        // winSize: Arama penceresi boyutu (21x21 iyidir)
        // maxLevel: Piramit seviyesi (büyük hareketler için 3 iyidir)
        calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err, Size(21, 21), 3); // bulunan köşeleri takip et ve ekrana çizdir

        // Matris hesabı için "sadece iyi noktaları" tutacak vektörler
        vector<Point2f> p_prev_good;
        vector<Point2f> p_curr_good;

        // 4. İyi noktaları seç ve çiz
        vector<Point2f> good_new; // Bir sonraki döngü için
        for (uint i = 0; i < prev_pts.size(); i++) {
            // Eğer nokta takibi başarısızsa atla
            // Eğer takip başarılıysa (status == 1)
            if (status[i] == 1) {
                // Matris hesabı için sakla
                p_prev_good.push_back(prev_pts[i]);
                p_curr_good.push_back(curr_pts[i]);

                // Bir sonraki frame için sakla
                good_new.push_back(curr_pts[i]);

                // ÇİZ
                // Hareket çizgisi çiz (Yeşil)
                line(mask, curr_pts[i], prev_pts[i], Scalar(0, 255, 0), 2);
                // Nokta koy (Kırmızı)
                circle(curr_frame, curr_pts[i], 5, Scalar(0, 0, 255), -1);
            }
        }


        // -----------------------------------------------------------------------
        //  HAREKET KESTİRİMİ (HAFTA 2)
        // -----------------------------------------------------------------------

        // Yeterli nokta varsa (örn. en az 10 nokta) matris hesapla
        if (p_prev_good.size() > 10) {
            // Rigid Transform (Kayma + Dönme) hesapla
            // false parametresi "full affine" değil, "partial affine" (shear yok) demektir.
            Mat T = estimateAffinePartial2D(p_prev_good, p_curr_good);

            if (!T.empty()) {
                // Matristen hareket verilerini çek
                // inter-frame (kareler arası hareket)
                // Bunları toplarsak (kümülatif toplam), kameranın başlangıçtan beri izlediği yolu buluruz.
                double dx = T.at<double>(0, 2); // Yatay kayma
                double dy = T.at<double>(1, 2); // Dikey kayma
                double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0)); // Dönme açısı (radyan)

                // Konsola yazdır (Doğrulama için)
                cout << "dx: " << dx << " | dy: " << dy << " | angle: " << da << endl;
                
                //Dosyaya kaydet
                // static int frame_idx = 0; frame_idx++; // Frame sayacı ekleyebilirsin
                fprintf(fp, "%d, %f, %f, %f\n", frame_idx, dx, dy, da); // i yerine kendi frame sayacını kullan
            }
        }
        // -----------------------------------------------------------------------


        // Sonucu göster
        Mat img;
        add(curr_frame, mask, img);

        // Oluşan kareyi dosyaya kaydet
        writer.write(img);

        // -------------------------------------------------------------
        // [MODÜLER GÖRÜNTÜLEME]
        // Pencereyi kapatmak (performans artışı) için aşağıdaki "if" bloğunu tamamen yorum satırı yap.
        // -------------------------------------------------------------
        
        //if (displayFrame(img)) { 
        //     break; // Fonksiyon true dönerse ('q' basıldıysa) döngüyü kır
        //}

        // -------------------------------------------------------------

        // 5. Bir sonraki döngü için 'şu anki' kareyi 'önceki' yap
        prev_gray = curr_gray.clone();
        prev_pts = good_new;

        // Nokta sayısı çok azaldıysa (örn. sahne değiştiyse) yeniden nokta bul
        if (prev_pts.size() < 10) {
            goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);
            mask = Mat::zeros(curr_frame.size(), curr_frame.type()); // Çizgileri temizle
        }
    }

    // Kaynakları serbest bırak
    cap.release();
    writer.release();
    destroyAllWindows();

    fclose(fp);

    return 0;
}