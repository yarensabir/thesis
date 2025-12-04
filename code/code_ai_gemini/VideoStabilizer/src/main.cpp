/*
Kodda Yaptığım Kritik Değişiklikler (Senin Koduna Kıyasla)
Döngüyü İkiye Böldüm:

Senin kodunda her kareyi okurken hemen writer.write yapıyorduk.

Bu yeni kodda ilk döngü (while) sadece veriyi topluyor (push_back), videoyu yazmıyor.

Video yazma işini en sondaki ikinci döngüye (for) bıraktık.

Veri Depolama:

vector<TransformParam> yapılarını ekledim. Çünkü geleceği görmeden (trajectory tamamlanmadan) smoothing yapamayız.

Çizimleri Kaldırdım:

line ve circle komutlarını kaldırdım. Çünkü artık amacımız "Yeşil çizgileri görmek" değil, "Stabilize olmuş temiz görüntüyü" elde etmek. Çizgiler stabilize videoda dikkat dağıtır.

Dosya Yolları:

output_videos yolunu güncelledim (Stabilized_Final.mp4).



Çift Geçiş (Two-Pass) yöntemi, offline (dosya tabanlı) işlemler içindir. Tüm videoyu bilirsen mükemmel sonuç alırsın ama canlı yayında (drone, güvenlik kamerası vb.) geleceği bilemezsin.

Gerçek zamanlı (Real-Time) stabilizasyon için kullanman gereken yöntem: KAYAN PENCERE (SLIDING WINDOW) veya ONE-PASS yöntemidir.

*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque> // Tampon için gerekli kütüphane
#include <cmath>

using namespace cv;
using namespace std;

// -----------------------------------------------------------------------------
// VERİ YAPILARI
// -----------------------------------------------------------------------------
struct TransformParam {
    double dx, dy, da;
    TransformParam() : dx(0), dy(0), da(0) {}
    TransformParam(double _dx, double _dy, double _da) : dx(_dx), dy(_dy), da(_da) {}

    TransformParam operator+(const TransformParam& p) const {
        return TransformParam(dx + p.dx, dy + p.dy, da + p.da);
    }
    TransformParam operator/(double val) const {
        return TransformParam(dx / val, dy / val, da / val);
    }
    TransformParam operator-(const TransformParam& p) const {
        return TransformParam(dx - p.dx, dy - p.dy, da - p.da);
    }
    TransformParam& operator+=(const TransformParam& rhs) {
        dx += rhs.dx;
        dy += rhs.dy;
        da += rhs.da;
        return *this;
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

// -----------------------------------------------------------------------------
// ANA PROGRAM (REAL-TIME ONE PASS)
// -----------------------------------------------------------------------------
int main() {
    // 1. AYARLAR
    string inputVideoPath = "/home/pi/Documents/thesis/test_video/ControlCam_7sec.mp4";
    // Eğer USB Kamera kullanacaksan: 
    // VideoCapture cap(0); 

    VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) { cerr << "Video acilamadi!" << endl; return -1; }

    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    if(fps <= 0) fps = 30; // Kamera fps vermezse varsayılan

    // Çıktı Videosu
    VideoWriter writer("/home/pi/Documents/thesis/code/code_ai_gemini/VideoStabilizer/outputs/041225/RealTime_Stabilized_1.mp4", 
                       VideoWriter::fourcc('m','p','4','v'), fps, Size(w,h));

    // 2. TAMPON AYARLARI (Buffer)
    // Ne kadar büyük olursa o kadar iyi smooth eder ama GECİKME artar.
    // 30 frame @ 30fps = 1 saniye gecikme
    const int BUFFER_SIZE = 30; 
    
    deque<Mat> frame_buffer;            // Ham görüntüleri tutar
    deque<TransformParam> motion_buffer; // O karelerin hareketini tutar
    TransformParam cumulative_motion(0,0,0); // Toplam yörünge

    Mat prev, curr, prev_gray, curr_gray;
    vector<Point2f> prev_pts, curr_pts;

    // İlk kareyi hazırlık
    cap >> prev;
    if(prev.empty()) return 0;
    cvtColor(prev, prev_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(prev_gray, prev_pts, 50, 0.01, 30);
    
    // Yörünge başlangıcı (0 noktası)
    motion_buffer.push_back(TransformParam(0,0,0));
    frame_buffer.push_back(prev.clone());

    cout << "--- REAL-TIME STABILIZER BASLIYOR ---" << endl;
    cout << "Tampon dolana kadar bekleyiniz (" << BUFFER_SIZE << " kare)..." << endl;

    int frame_idx = 0;

    while(true) {
        // A. YENİ KAREYİ OKU VE HAREKETİ HESAPLA
        cap >> curr;
        if(curr.empty()) break; // Kamera veya video bitti

        cvtColor(curr, curr_gray, COLOR_BGR2GRAY);

        // Optical Flow
        vector<uchar> status;
        vector<float> err;
        if(prev_pts.size() > 0) {
            calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err, Size(21, 21), 3);
        }

        // İyi noktaları ayıkla
        vector<Point2f> p0, p1;
        for(size_t i=0; i<status.size(); i++) {
            if(status[i]) { p0.push_back(prev_pts[i]); p1.push_back(curr_pts[i]); }
        }

        // Hareket Matrisi
        double dx=0, dy=0, da=0;
        if(p0.size() > 10) {
            Mat T = estimateAffinePartial2D(p0, p1);
            if(!T.empty()) {
                dx = T.at<double>(0,2);
                dy = T.at<double>(1,2);
                da = atan2(T.at<double>(1,0), T.at<double>(0,0));
            }
        }

        // Kümülatif Yörüngeyi Güncelle
        cumulative_motion += TransformParam(dx, dy, da);
        
        // Verileri Tampona At
        motion_buffer.push_back(cumulative_motion);
        frame_buffer.push_back(curr.clone());

        // B. TAMPON DOLDUYSA İŞLE VE ÇIKTI ÜRET
        if(frame_buffer.size() > BUFFER_SIZE) {
            
            // 1. Yumuşatma (Smoothing) - Tamponun ortalamasını al
            TransformParam sum(0,0,0);
            for(const auto& m : motion_buffer) {
                sum += m;
            }
            TransformParam smoothed_pos = sum / (double)motion_buffer.size();

            // 2. İşlenecek kare: Tamponun en başındaki (en eski) kare
            Mat target_frame = frame_buffer.front(); 
            TransformParam target_pos = motion_buffer.front(); // O karenin orijinal konumu

            // 3. Farkı Hesapla (Smooth - Original)
            //TransformParam diff = smoothed_pos - target_pos;
            TransformParam diff = target_pos - smoothed_pos;

            // 4. Warping
            double warp_dx = diff.dx;
            double warp_dy = diff.dy;
            double warp_da = diff.da;

            Mat M = Mat::eye(2, 3, CV_64F);
            M.at<double>(0,0) = cos(warp_da);
            M.at<double>(0,1) = -sin(warp_da);
            M.at<double>(1,0) = sin(warp_da);
            M.at<double>(1,1) = cos(warp_da);
            M.at<double>(0,2) = warp_dx;
            M.at<double>(1,2) = warp_dy;

            Mat stabilized;
            warpAffine(target_frame, stabilized, M, target_frame.size());

            // 5. Çıktı (Diske veya Ekrana)
            writer.write(stabilized);
            // imshow("Real-Time Stabilizer", stabilized);
            // if(waitKey(1) == 'q') break;

            // 6. Tampondan eskileri at
            frame_buffer.pop_front();
            motion_buffer.pop_front();
            
            if(frame_idx % 30 == 0) cout << "Processing Frame: " << frame_idx << endl;
        }

        // C. HAZIRLIKLAR
        prev_gray = curr_gray.clone();
        prev_pts = p1;
        if(prev_pts.size() < 10) goodFeaturesToTrack(prev_gray, prev_pts, 50, 0.01, 30);
        
        frame_idx++;
    }

    // Tamponda kalan son kareleri de boşalt (Video bittiğinde)
    while(!frame_buffer.empty()) {
        writer.write(frame_buffer.front());
        frame_buffer.pop_front();
    }

    cout << "Islem Bitti." << endl;
    return 0;
}


/*
Döngü Yapısı: Kod artık tek bir while döngüsü içinde hem okuma hem yazma yapıyor.

deque (Buffer): Görüntüleri hemen işlemiyoruz. BUFFER_SIZE (30 kare) kadar biriktiriyoruz.

İlk 30 kare boyunca ekrana/dosyaya hiçbir şey çıkmaz (Bu gecikme süresidir).

kare kamera tarafından çekildiğinde, biz 1. kareyi işleyip dışarı atarız.

Bu sayede 1. kareyi işlerken, gelecekteki 30 karenin nereye gittiğini biliyoruz!

RAM Kullanımı: frame_buffer içinde 30 tane resim tutuyoruz. Raspberry Pi için bu sorun değil (yaklaşık 50-100MB RAM harcar), ama BUFFER_SIZE değerini 1000 yaparsan RAM yetmeyebilir.

Bu yöntem, profesyonel yayıncılıkta kullanılan "Canlı yayın geciktiricisi" mantığının aynısıdır.
*/