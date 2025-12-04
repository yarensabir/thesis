#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Orijinal videonun yolu
    std::string inputPath = "/home/pi/Documents/thesis/test_video/ControlCam_20200930_104820.mp4"; // Dosya adınızı buraya yazın
    std::string outputPath = "/home/pi/Documents/thesis/test_video/ControlCam_7sec.mp4";

    // Video yakalama nesnesini oluştur
    cv::VideoCapture cap(inputPath);

    // Videonun açılıp açılmadığını kontrol et
    if (!cap.isOpened()) {
        std::cerr << "Hata: Video dosyası açılamadı!" << std::endl;
        return -1;
    }

    // 2. Video özelliklerini al
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Toplam kaç kare kaydedileceğini hesapla (3 saniye x FPS)
    int maxFrames = static_cast<int>(fps * 7);

    std::cout << "Video FPS: " << fps << std::endl;
    std::cout << "Hedeflenen Kare Sayisi (1 sn): " << maxFrames << std::endl;

    // 3. Video yazıcıyı (VideoWriter) hazırla
    // MP4 formatı için 'mp4v' codec'i kullanılır.
    cv::VideoWriter writer(outputPath, 
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           fps, 
                           cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Hata: Video yazıcı başlatılamadı!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frameCount = 0;

    // 4. Kareleri oku ve yaz
    while (true) {
        cap >> frame; // Sıradaki kareyi oku

        if (frame.empty()) {
            std::cout << "Video sonuna gelindi." << std::endl;
            break;
        }

        // 3 saniyelik limit dolduysa döngüden çık
        if (frameCount >= maxFrames) {
            std::cout << "3 saniyelik kısım kaydedildi." << std::endl;
            break;
        }

        writer.write(frame); // Kareyi yeni dosyaya yaz
        frameCount++;
        
        // İlerleme durumunu göster (isteğe bağlı)
        if (frameCount % 10 == 0) {
             std::cout << "İşlenen kare: " << frameCount << "/" << maxFrames << "\r" << std::flush;
        }
    }

    // 5. Kaynakları serbest bırak
    cap.release();
    writer.release();

    std::cout << "\nİşlem tamamlandı! Dosya: " << outputPath << std::endl;

    return 0;
}