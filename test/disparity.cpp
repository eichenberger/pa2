#include <vector>
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/ximgproc.hpp"

#include "xunit_lib_tara.h"

using namespace cv;
using namespace std;


void mouseHandler(int event, int x, int y, int flags, void *userData)
{
    Mat *img = static_cast<Mat*>(userData);

    if (event == EVENT_LBUTTONDOWN)
        printf("Distance at %dx%d: %f\n", x, y, img->at<double>(y, x));
}

int main(int argc, char **argv)
{
    int key = 0;
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;
    cout << "Use OCL: " << cv::ocl::useOpenCL() << endl;
    VideoCapture cap(argv[1]);
    cap.set(CV_CAP_PROP_FRAME_WIDTH,752);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

    cout << "Read settings" << endl;
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cout << "Parse settings" << endl;

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify left_matcher are missing!" << endl;
        return -1;
    }

    cout << "Set autoexposure" << endl;
    InitExtensionUnit(const_cast<char*>("test"));
    SetAutoExposureStereo();

    cout << "Start program" << endl;
    cv::Mat M1l,M2l,M1r,M2r;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);

    Ptr<StereoSGBM> left_matcher = StereoSGBM::create();

    namedWindow("Left", WINDOW_AUTOSIZE | CV_GUI_EXPANDED);
    namedWindow("Right", WINDOW_AUTOSIZE | CV_GUI_EXPANDED);

    unsigned int sgbm_preFilterCap	=	61;
    unsigned int sgbm_SADWindowSize	=	8;
    unsigned int sgbm_minDisparity	=	0;
    unsigned int sgbm_speckleRange	=	31;
    unsigned int sgbm_disp12MaxDiff	=	1;
    unsigned int sgbm_uniquenessRatio	 =	0;
    unsigned int sgbm_speckleWindowSize	 =	200;
    unsigned int sgbm_numberOfDisparities =	64;
    int cn = 1;

    left_matcher->setPreFilterCap(sgbm_preFilterCap);
    left_matcher->setBlockSize (sgbm_SADWindowSize > 0 ? sgbm_SADWindowSize : 3);
    left_matcher->setP1(8 * cn * sgbm_SADWindowSize * sgbm_SADWindowSize);
    left_matcher->setP2(32 * cn * sgbm_SADWindowSize * sgbm_SADWindowSize);
    left_matcher->setNumDisparities(sgbm_numberOfDisparities);
    left_matcher->setMinDisparity(sgbm_minDisparity);
    left_matcher->setUniquenessRatio(sgbm_uniquenessRatio);
    left_matcher->setSpeckleWindowSize(sgbm_speckleWindowSize);
    left_matcher->setSpeckleRange(sgbm_speckleRange);
    left_matcher->setDisp12MaxDiff(sgbm_disp12MaxDiff);

    left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);


    Ptr<StereoMatcher> right_matcher = ximgproc::createRightMatcher(left_matcher.dynamicCast<StereoMatcher>());
    Ptr<ximgproc::DisparityWLSFilter> wls_filter = ximgproc::createDisparityWLSFilter(left_matcher);
    Mat depth;
    namedWindow("Depth");
    namedWindow("Left");
    cv::setMouseCallback("Depth", mouseHandler, &depth);
    cv::setMouseCallback("Left", mouseHandler, &depth);

    while (key != 'q'){
        Mat img, imLeft, imRight, disparityL, disparityR, imLeftRect, imRightRect, disparityFiltered;
        cap.read(img);

        extractChannel(img, imLeft, 1);
        extractChannel(img, imRight, 2);

        cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);

        left_matcher->compute(imLeftRect, imRightRect, disparityL);
        right_matcher->compute(imRightRect, imLeftRect, disparityR);

        wls_filter->setLambda(4000.0);
        wls_filter->setSigmaColor(1.0);
        wls_filter->filter(disparityL, imLeftRect, disparityFiltered, disparityR);

        printf("Calculate depth\n");
        Mat disparityFractional;
        disparityFiltered.convertTo(disparityFractional, CV_64F, 1.0);
        // z = b*f/d
        float bf = fsSettings["Camera.bf"];
        if (bf == 0)
            bf = 451.932;

        depth = bf/(disparityFractional);

        cv::normalize(disparityFiltered, disparityFiltered, 0, 255, CV_MINMAX, CV_8U);
        cv::applyColorMap(disparityFiltered, disparityFiltered, cv::COLORMAP_JET);

        imshow("Left", imLeftRect);
        imshow("Right", imRightRect);
        imshow("Disparity left", disparityFiltered);
        imshow("Depth", depth);
        key = waitKey(10);
    }

    return 0;
}
