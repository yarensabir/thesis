#include "Transform.h"
#include <cmath>

TransformParam::TransformParam() : dx(0), dy(0), da(0) {}
TransformParam::TransformParam(double _dx, double _dy, double _da) : dx(_dx), dy(_dy), da(_da) {}

TransformParam TransformParam::operator+(const TransformParam& p) const {
    return TransformParam(dx + p.dx, dy + p.dy, da + p.da);
}
TransformParam TransformParam::operator-(const TransformParam& p) const {
    return TransformParam(dx - p.dx, dy - p.dy, da - p.da);
}
TransformParam TransformParam::operator/(double val) const {
    return TransformParam(dx / val, dy / val, da / val);
}
void TransformParam::operator+=(const TransformParam& p) {
    dx += p.dx; dy += p.dy; da += p.da;
}

cv::Mat getAffineFromParam(TransformParam p) {
    cv::Mat m = cv::Mat::eye(2, 3, CV_64F);
    m.at<double>(0,0) = cos(p.da);
    m.at<double>(0,1) = -sin(p.da);
    m.at<double>(1,0) = sin(p.da);
    m.at<double>(1,1) = cos(p.da);
    m.at<double>(0,2) = p.dx;
    m.at<double>(1,2) = p.dy;
    return m;
}