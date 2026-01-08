#pragma once
#include <string>

enum RealTimeMethod {
    RT_SLIDING_WINDOW = 1,
    RT_KALMAN_FILTER = 2
};

enum OfflineMethod {
    OFF_MOVING_AVERAGE = 1,
    OFF_GAUSSIAN = 2
};

// DEĞİŞİKLİK BURADA: Sona 'std::string outputName' ekledik
void runRealTimeStabilization(RealTimeMethod method, std::string outputName);
void runOfflineStabilization(std::string inputPath, OfflineMethod method, std::string outputName);