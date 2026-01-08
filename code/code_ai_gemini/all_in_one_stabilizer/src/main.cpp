#include <iostream>
#include "Stabilizer.h" 

using namespace std;

int main() {
    string videoPath = "/home/pi/Documents/thesis/test_video/ControlCam_7sec.mp4"; 

    int mainChoice, subChoice;
    cout << "==========================================" << endl;
    cout << "VIDEO STABILIZATION SYSTEM" << endl;
    cout << "==========================================" << endl;
    cout << "1. Real-Time Stabilization (Pi Camera)" << endl;
    cout << "2. Offline Stabilization (Video File)" << endl;
    cout << "Secim: ";
    cin >> mainChoice;

    if (mainChoice == 1) {
        cout << "\n--- Real-Time Yontem Secimi ---" << endl;
        cout << "1. Sliding Window (Buffer Average)" << endl;
        cout << "2. Kalman Filter" << endl;
        cout << "Yontem: ";
        cin >> subChoice;

        if(subChoice == 1) runRealTimeStabilization(RT_SLIDING_WINDOW);
        else if(subChoice == 2) runRealTimeStabilization(RT_KALMAN_FILTER);
        else cout << "Hatali secim!" << endl;

    } else if (mainChoice == 2) {
        cout << "\n--- Offline Yontem Secimi ---" << endl;
        cout << "1. Moving Average (Basit)" << endl;
        cout << "2. Gaussian Smoothing (Sinematik)" << endl;
        cout << "Yontem: ";
        cin >> subChoice;

        if(subChoice == 1) runOfflineStabilization(videoPath, OFF_MOVING_AVERAGE);
        else if(subChoice == 2) runOfflineStabilization(videoPath, OFF_GAUSSIAN);
        else cout << "Hatali secim!" << endl;

    } else {
        cout << "Gecersiz ana secim!" << endl;
    }

    return 0;
}