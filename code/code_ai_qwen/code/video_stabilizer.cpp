#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Video dosyasının yolu
    string video_path = "../s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820.mp4";

    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Hata: Video açılamadı: " << video_path << endl;
        return -1;
    }

    Mat prev, curr, prev_gray, curr_gray;
    cap >> prev;
    cvtColor(prev, prev_gray, COLOR_BGR2GRAY);

    vector<Point2f> prev_points, curr_points;
    vector<uchar> status;
    vector<float> err;

    // Shi-Tomasi köşe detektörü parametreleri
    TermCriteria term_criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
    int max_corners = 300;
    double quality_level = 0.01;
    double min_distance = 30;

    // VideoWriter ile stabilize edilmiş videoyu yaz
    VideoWriter out("stabilized_output.avi",
                    VideoWriter::fourcc('M','J','P','G'),
                    cap.get(CAP_PROP_FPS),
                    Size(prev.cols, prev.rows));

    cout << "Video stabilizasyonu başlıyor..." << endl;

    while (true) {
        cap >> curr;
        if (curr.empty()) break;

        cvtColor(curr, curr_gray, COLOR_BGR2GRAY);

        // Köşe noktalarını bul
        goodFeaturesToTrack(prev_gray, prev_points, max_corners, quality_level, min_distance);

        if (prev_points.size() > 0) {
            // Lucas-Kanade optik akış ile noktaları takip et
            calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, curr_points, status, err);

            // Sadece başarılı takip edilen noktaları filtrele
            vector<Point2f> prev_filtered, curr_filtered;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i]) {
                    prev_filtered.push_back(prev_points[i]);
                    curr_filtered.push_back(curr_points[i]);
                }
            }

            if (prev_filtered.size() >= 3) {
                // Affine dönüşümü hesapla
                Mat transform = estimateAffinePartial2D(prev_filtered, curr_filtered);

                if (transform.empty()) {
                    transform = Mat::eye(2, 3, CV_64F); // Hata durumunda birim matris
                }

                // Dönüşümü tersine çevir (stabilizasyon için)
                Mat inv_transform = Mat::eye(3, 3, CV_64F);
                transform.copyTo(inv_transform(Rect(0, 0, 3, 2)));
                inv_transform = inv_transform.inv();

                // Stabilize edilmiş kareyi oluştur
                Mat stabilized;
                warpAffine(curr, stabilized, inv_transform(Rect(0, 0, 3, 2)), curr.size());

                // Çıktıya yaz
                out.write(stabilized);
            } else {
                out.write(curr); // Eğer yeterli nokta yoksa orijinal kareyi yaz
            }
        } else {
            out.write(curr);
        }

        curr.copyTo(prev);
        curr_gray.copyTo(prev_gray);
    }

    cap.release();
    out.release();
    cout << "Video stabilizasyonu tamamlandı. Çıktı: stabilized_output.avi" << endl;
    return 0;
}
