#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

using namespace std;
using namespace cv;

namespace {
    const char* about = "Pose estimation of ArUco marker images";
    const char* keys  =
            "{d        |16    | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, "
            "DICT_4X4_250=2, DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, "
            "DICT_5X5_250=6, DICT_5X5_1000=7, DICT_6X6_50=8, DICT_6X6_100=9, "
            "DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12, DICT_7X7_100=13, "
            "DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
            "{h        |false | Print help }"
            "{l        |0.247 | Actual marker length in meter }"
    ;
}

void computeC2MC1(Mat &R1, Mat &tvec1, Mat &R2, Mat &tvec2,
                  Mat &R_1to2, Mat &tvec_1to2)
{
    //c2Mc1 = c2Mo * oMc1 = c2Mo * c1Mo.inv()

    cv::Mat T1;
    tvec1.convertTo(T1, CV_64F);

    cv::Mat T2;
    tvec2.convertTo(T2, CV_64F);

    cv::Mat r1;
    R1.convertTo(r1, CV_64F);

    cv::Mat r2;
    R2.convertTo(r2, CV_64F);

    R_1to2 = r2 * r1.t();
    tvec_1to2 = r2 * (-r1.t()*T1) + T2;
}

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if (argc < 1) {
        parser.printMessage();
        return 1;
    }

    if (parser.get<bool>("h")) {
        parser.printMessage();
        return 0;
    }

    int dictionaryId = parser.get<int>("d");
    float marker_length_m = parser.get<float>("l");
    int wait_time = 10;

    if (marker_length_m <= 0) {
        std::cerr << "marker length must be a positive value in meter"
                  << std::endl;
        return 1;
    }

    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }

//    cv::Mat camera_matrix1, dist_coeffs1;
//    cv::Mat camera_matrix2, dist_coeffs2;

    cv::Mat camera_matrix1 = cv::Mat_<float>(3, 3);
    camera_matrix1.at<float>(0, 0)=  1053;
    camera_matrix1.at<float>(0, 1)=   0.0;
    camera_matrix1.at<float>(0, 2)=  71.8;
    camera_matrix1.at<float>(1, 0)=  0.0;
    camera_matrix1.at<float>(1, 1)=  1053;
    camera_matrix1.at<float>(1, 2)=  546.394;
    camera_matrix1.at<float>(2, 0)=  0.0;
    camera_matrix1.at<float>(2, 1)=  0.0;
    camera_matrix1.at<float>(2, 2)=  1.0;

    cv::Mat dist_coeffs1 = cv::Mat_<float>(1, 5);
    dist_coeffs1.at<float>(0, 0)=  0.0;
    dist_coeffs1.at<float>(0, 1)=  0.0;
    dist_coeffs1.at<float>(0, 2)=  0.0;
    dist_coeffs1.at<float>(0, 3)=  0.0;
    dist_coeffs1.at<float>(0, 4)=  0.0;

    cv::Mat camera_matrix2 = cv::Mat_<float>(3, 3);
    camera_matrix2.at<float>(0, 0)=  605.839599609375;
    camera_matrix2.at<float>(0, 1)=   0.0;
    camera_matrix2.at<float>(0, 2)=  315.699493408203;
    camera_matrix2.at<float>(1, 0)=  0.0;
    camera_matrix2.at<float>(1, 1)=  606.218078613281;
    camera_matrix2.at<float>(1, 2)=  241.644165039062;
    camera_matrix2.at<float>(2, 0)=  0.0;
    camera_matrix2.at<float>(2, 1)=  0.0;
    camera_matrix2.at<float>(2, 2)=  1.0;

    cv::Mat dist_coeffs2 = cv::Mat_<float>(1, 5);
    dist_coeffs2.at<float>(0, 0)=  0.0;
    dist_coeffs2.at<float>(0, 1)=  0.0;
    dist_coeffs2.at<float>(0, 2)=  0.0;
    dist_coeffs2.at<float>(0, 3)=  0.0;
    dist_coeffs2.at<float>(0, 4)=  0.0;


    cv::Ptr<cv::aruco::Dictionary> dictionary =
            cv::aruco::getPredefinedDictionary( \
        cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

//    cv::FileStorage fs1("center_camera.yml", cv::FileStorage::READ);
//    fs1["camera_matrix"] >> camera_matrix1;
//    fs1["distortion_coefficients"] >> dist_coeffs1;

    std::cout << "Camera 1: \n";
    std::cout << "camera_matrix\n" << camera_matrix1 << std::endl;
    std::cout << "dist coeffs\n" << dist_coeffs1 << std::endl;

//    cv::FileStorage fs2("left_camera.yml", cv::FileStorage::READ);
//    fs2["camera_matrix"] >> camera_matrix2;
//    fs2["distortion_coefficients"] >> dist_coeffs2;

    std::cout << "Camera 2: \n";
    std::cout << "camera_matrix\n" << camera_matrix2 << std::endl;
    std::cout << "dist coeffs\n" << dist_coeffs2 << std::endl;

    Mat img1 = imread( samples::findFile("/home/jun/Github/ComputerVisionTools/ArUcO_Example/imgs/center-left/center.jpg"));
    Mat img2 = imread( samples::findFile("/home/jun/Github/ComputerVisionTools/ArUcO_Example/imgs/center-left/left.jpg"));

    std::vector<int> ids1, ids2;
    std::vector<std::vector<cv::Point2f> > corners1, corners2;

    cv::aruco::detectMarkers(img1, dictionary, corners1, ids1);
    cv::aruco::detectMarkers(img2, dictionary, corners2, ids2);

    cv::Mat rvecs1, tvecs1;
    cv::Mat rvecs2, tvecs2;

    // Camera 1 processing
    if (ids1.size() > 0)
    {
        cv::aruco::drawDetectedMarkers(img1, corners1, ids1);
        cv::aruco::estimatePoseSingleMarkers(corners1, marker_length_m,
                                                 camera_matrix1, dist_coeffs1, rvecs1, tvecs1);

        std::cout << "Translation: " << tvecs1
                      << "\nRotation: " << rvecs1
                      << std::endl;

        cv::aruco::drawAxis(img1, camera_matrix1, dist_coeffs1,
                            rvecs1, tvecs1, 0.1);

        imshow("Pose estimation", img1);
        cv::waitKey(0);
    }

    // Camera 2 processing
    if (ids2.size() > 0)
    {
        cv::aruco::drawDetectedMarkers(img2, corners2, ids2);
        cv::aruco::estimatePoseSingleMarkers(corners2, marker_length_m,
                                             camera_matrix2, dist_coeffs2, rvecs2, tvecs2);

        std::cout << "Translation: " << tvecs2
                  << "\nRotation: " << rvecs2
                  << std::endl;

        cv::aruco::drawAxis(img2, camera_matrix2, dist_coeffs2,
                            rvecs2, tvecs2, 0.1);

        imshow("Pose estimation", img2);
        cv::waitKey(0);
    }

    // Compute transformation matrix 2 to 1
    Mat R1, R2;
    Rodrigues(rvecs1, R1);
    Rodrigues(rvecs2, R2);

    cout << R1 << "\n";
    cout << R2 << "\n";

    Mat R_1to2(1, 3, CV_64F);
    Mat t_1to2(1, 3, CV_64F);

    computeC2MC1(R1, tvecs1, R2, tvecs2, R_1to2, t_1to2);

    Mat rvec_1to2;
    Rodrigues(R_1to2, rvec_1to2);

    std::cout << R_1to2 << std::endl;
    std::cout << t_1to2 << std::endl;
    std::cout << rvec_1to2 << std::endl;

    return 0;
}