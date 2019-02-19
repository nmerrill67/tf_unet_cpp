#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>



// >>>> Kalman Filter
int stateSize = 6;
int measSize = 4;
int contrSize = 0;


// Main KF object
cv::KalmanFilter kf;

// Our state and measurement object
cv::Mat state;
cv::Mat state_old;
cv::Mat meas;

// last time this was tracks
double lastTimeStep = 0;

// Number of seconds that if passed, means we lost the track
double dt_threshold = 5.0;






// All this code has been taken from this blog post here:
// https://www.myzhar.com/blog/tutorials/tutorial-opencv-ball-tracker-using-kalman-filter/
void initialize_kf() {


    // init the kalman filter
    kf = cv::KalmanFilter(stateSize, measSize, contrSize, CV_32F);


    //cv::Mat procNoise(stateSize, 1, type)
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]


    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]


    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]

    kf = cv::KalmanFilter(6, 4, 0, CV_32F);

    state = cv::Mat(6, 1, CV_32F);  // [x,y,v_x,v_y,w,h]
    meas = cv::Mat(4, 1, CV_32F);
    lastTimeStep = 0;

    cv::setIdentity(kf.transitionMatrix);

    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    // kf.processNoiseCov.at<float>(0) = 1.0f;
    // kf.processNoiseCov.at<float>(7) = 1.0f;
    // kf.processNoiseCov.at<float>(14) = 5.0f;
    // kf.processNoiseCov.at<float>(21) = 5.0f;
    // kf.processNoiseCov.at<float>(28) = 1.0f;
    // kf.processNoiseCov.at<float>(35) = 1.0f;
    kf.processNoiseCov.at<float>(0) = 1e-1;
    kf.processNoiseCov.at<float>(7) = 1e-1;
    kf.processNoiseCov.at<float>(14) = 1.0f;
    kf.processNoiseCov.at<float>(21) = 1.0f;
    kf.processNoiseCov.at<float>(28) = 1e-1;
    kf.processNoiseCov.at<float>(35) = 1e-1;

    // Measures Noise Covariance Matrix R
    // cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(10));

}



// This will update the kalman filter given a new observation
// If we have lost track of it, then we should re-init the filter
bool update_kf(double newTimeStep, cv::Rect& measBB, cv::Rect& upRect) {

    // Calculate delta time
    double dTd = newTimeStep - lastTimeStep;
    float dT = (float) dTd;

    //std::cout << "dT- " << dT << std::endl << std::endl << std::endl;

    // >>>> Matrix A
    kf.transitionMatrix.at<float>(2) = dT;
    kf.transitionMatrix.at<float>(9) = dT;
    // <<<< Matrix A

    // Predict forward in time
    state_old = state.clone();
    state = kf.predict();

    // <<<<< Noise smoothing
    // <<<<< Detection result
    meas.at<float>(0) = measBB.x + measBB.width / 2;
    meas.at<float>(1) = measBB.y + measBB.height / 2;
    meas.at<float>(2) = (float) measBB.width;
    meas.at<float>(3) = (float) measBB.height;

    // Track if we did an update or not
    bool did_update = false;

    // std::cout  << "dt = " << dTd << std::endl;

    // If lost or initial track, then init it!
    if(dTd > dt_threshold) {

        std::cout  << std::endl << "initalizing KF - dt = " << dTd << std::endl;

        // >>>> Initialization
        kf.errorCovPre.at<float>(0) = 1; // px
        kf.errorCovPre.at<float>(7) = 1; // px
        kf.errorCovPre.at<float>(14) = 1;
        kf.errorCovPre.at<float>(21) = 1;
        kf.errorCovPre.at<float>(28) = 1; // px
        kf.errorCovPre.at<float>(35) = 1; // px

        state.at<float>(0) = meas.at<float>(0);
        state.at<float>(1) = meas.at<float>(1);
        state.at<float>(2) = 0;
        state.at<float>(3) = 0;
        state.at<float>(4) = meas.at<float>(2);
        state.at<float>(5) = meas.at<float>(3);
        // <<<< Initialization
        kf.statePost = state;
        lastTimeStep = newTimeStep;

    } else {

        // compute the error in this measurement
        cv::Mat err = meas - kf.measurementMatrix*state;
        // std::cout << "kf error = " << err << std::endl;

        // calc state covariance if this measurement is to be included
        cv::Mat S = kf.measurementMatrix*kf.errorCovPre*kf.measurementMatrix.t() + kf.measurementNoiseCov;
        cv::Mat S_inv = S.inv();

        // calc Mahalanobis distance
        cv::Mat chi = err.t()*S_inv*err;
        // std::cout << "kf error = " << err << std::endl;
        std::cout << "chi2 = " << chi.at<float>(0) << std::endl;

        if(chi.at<float>(0) < 200) {
            kf.correct(meas);
            lastTimeStep = newTimeStep;
            did_update = true;
        }
    }


    // Record this retangle
    upRect.width = state.at<float>(4);
    upRect.height = state.at<float>(5);
    upRect.x = state.at<float>(0) - upRect.width / 2;
    upRect.y = state.at<float>(1) - upRect.height / 2;
    return did_update;

}








