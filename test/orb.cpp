#include <vector>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    int key = 0;
    double t_detect = 0;
    double t_compute = 0;
    int i = 0;
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;
    cout << "Use OCL: " << cv::ocl::useOpenCL() << endl;
    VideoCapture cap(argv[1]);
    Ptr<ORB> orb = ORB::create(1200, 1.2, 8);
    while (key != 'q' && i < 500){
        Mat img;
        UMat uimg;
        cap.read(img);
        Mat gray;
        UMat uGray;
        int64 e1, e2, e3;
        Mat descriptors;
        UMat uDescriptors;
        std::vector<KeyPoint> keypoints;

        extractChannel(img, gray, 1);
        //cvtColor(img, gray, COLOR_RGB2GRAY);

        //e1 = getTickCount();
        //orb->detect(gray, keypoints);
        //e2 = getTickCount();
        //gray.copyTo(uGray);
        //orb->compute(uGray, keypoints, descriptors);
        //e3 = getTickCount();

        e1 = getTickCount();
        gray.copyTo(uGray);
        e2 = getTickCount();
        orb->detectAndCompute(uGray, Mat(), keypoints, uDescriptors);
        e3 = getTickCount();

//        e1 = getTickCount();
//        orb->detectAndCompute(gray, Mat(), keypoints, descriptors);
//        e2 = getTickCount();

//        gray.copyTo(uGray);
//        orb->compute(uGray, keypoints, descriptors);
//        e3 = getTickCount();


        drawKeypoints(gray, keypoints, gray);
        imshow("image", gray);
        key = waitKey(1);

        t_detect += e2-e1;
        t_compute += e3-e2;
        i++;
    }

    cout << "Detect: " << (t_detect)/(getTickFrequency()*i) << endl;
    cout << "Compute: " << (t_compute)/(getTickFrequency()*i) << endl;

    return 0;
}
