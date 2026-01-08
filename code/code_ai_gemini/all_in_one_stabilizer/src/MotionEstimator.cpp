#include "MotionEstimator.h"

using namespace cv;
using namespace std;

MotionEstimator::MotionEstimator() {
    max_corners = 50;
    quality = 0.01;
    min_dist = 30;
}

void MotionEstimator::initialize(Mat& first_frame) {
    cvtColor(first_frame, prev_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(prev_gray, prev_pts, max_corners, quality, min_dist);
}

TransformParam MotionEstimator::estimate(Mat& curr_frame) {
    Mat curr_gray;
    cvtColor(curr_frame, curr_gray, COLOR_BGR2GRAY);

    vector<Point2f> curr_pts;
    vector<uchar> status;
    vector<float> err;
    
    if (prev_pts.size() > 0) {
        calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err, Size(21, 21), 3);
    }

    vector<Point2f> p0, p1;
    for(size_t i=0; i < status.size(); i++) {
        if(status[i]) {
            p0.push_back(prev_pts[i]);
            p1.push_back(curr_pts[i]);
        }
    }

    double dx=0, dy=0, da=0;
    if(p0.size() > 10) {
        Mat T = estimateAffinePartial2D(p0, p1);
        if(!T.empty()) {
            dx = T.at<double>(0,2);
            dy = T.at<double>(1,2);
            da = atan2(T.at<double>(1,0), T.at<double>(0,0));
        }
    }

    prev_gray = curr_gray.clone();
    prev_pts = p1;
    if(prev_pts.size() < 15) {
        goodFeaturesToTrack(prev_gray, prev_pts, max_corners, quality, min_dist);
    }

    return TransformParam(dx, dy, da);
}