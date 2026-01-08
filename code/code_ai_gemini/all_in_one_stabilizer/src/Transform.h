#pragma once
#include <opencv2/opencv.hpp>

// Veri yapısı tanımı
struct TransformParam {
    double dx, dy, da;

    TransformParam();
    TransformParam(double _dx, double _dy, double _da);

    TransformParam operator+(const TransformParam& p) const;
    TransformParam operator-(const TransformParam& p) const;
    TransformParam operator/(double val) const;
    void operator+=(const TransformParam& p);
};

// Yardımcı fonksiyon tanımı
cv::Mat getAffineFromParam(TransformParam p);