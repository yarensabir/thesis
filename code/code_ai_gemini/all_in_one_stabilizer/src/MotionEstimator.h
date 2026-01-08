#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "Transform.h" // Transform yapısını bilmesi gerek

class MotionEstimator {
private:
    cv::Mat prev_gray;
    std::vector<cv::Point2f> prev_pts;
    int max_corners;
    double quality;
    double min_dist;

public:
    MotionEstimator(); // Constructor
    void initialize(cv::Mat& first_frame);
    TransformParam estimate(cv::Mat& curr_frame);
};