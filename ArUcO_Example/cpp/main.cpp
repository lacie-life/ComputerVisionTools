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
    tvec1.convertTo(tvec1, CV_32FC1);
    tvec2.convertTo(tvec2, CV_32FC1);
    R1.convertTo(R1, CV_32FC1);
    R2.convertTo(R2, CV_32FC1);

    R_1to2 = R2 * R1.t();
    tvec_1to2 = R2 * (-R1.t()*tvec1) + tvec2;
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

    cv::Mat camera_matrix1 = cv::Mat_<double>(3, 3);
    camera_matrix1.at<double>(0, 0)=  1053;
    camera_matrix1.at<double>(0, 1)=   0.0;
    camera_matrix1.at<double>(0, 2)=  71.8;
    camera_matrix1.at<double>(1, 0)=  0.0;
    camera_matrix1.at<double>(1, 1)=  1053;
    camera_matrix1.at<double>(1, 2)=  546.394;
    camera_matrix1.at<double>(2, 0)=  0.0;
    camera_matrix1.at<double>(2, 1)=  0.0;
    camera_matrix1.at<double>(2, 2)=  1.0;

    cv::Mat dist_coeffs1 = cv::Mat_<double>(1, 5);
    dist_coeffs1.at<double>(0, 0)=  0.0;
    dist_coeffs1.at<double>(0, 1)=  0.0;
    dist_coeffs1.at<double>(0, 2)=  0.0;
    dist_coeffs1.at<double>(0, 3)=  0.0;
    dist_coeffs1.at<double>(0, 4)=  0.0;

    cv::Mat camera_matrix2 = cv::Mat_<double>(3, 3);
    camera_matrix2.at<double>(0, 0)=  607.157348632812;
    camera_matrix2.at<double>(0, 1)=   0.0;
    camera_matrix2.at<double>(0, 2)=  315.798370361328;
    camera_matrix2.at<double>(1, 0)=  0.0;
    camera_matrix2.at<double>(1, 1)=  607.830505371094;
    camera_matrix2.at<double>(1, 2)=  226.742065429688;
    camera_matrix2.at<double>(2, 0)=  0.0;
    camera_matrix2.at<double>(2, 1)=  0.0;
    camera_matrix2.at<double>(2, 2)=  1.0;

    cv::Mat dist_coeffs2 = cv::Mat_<double>(1, 5);
    dist_coeffs2.at<double>(0, 0)=  0.0;
    dist_coeffs2.at<double>(0, 1)=  0.0;
    dist_coeffs2.at<double>(0, 2)=  0.0;
    dist_coeffs2.at<double>(0, 3)=  0.0;
    dist_coeffs2.at<double>(0, 4)=  0.0;


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

    Mat img1 = imread( samples::findFile("/home/lacie/Github/ComputerVisionTools/ArUcO_Example/imgs/right2center/right.jpg"));
    Mat img2 = imread( samples::findFile("/home/lacie/Github/ComputerVisionTools/ArUcO_Example/imgs/right2center/center.jpg"));

    std::vector<int> ids1, ids2;
    std::vector<std::vector<cv::Point2f>> corners1, corners2;

    cv::aruco::detectMarkers(img1, dictionary, corners1, ids1);
    cv::aruco::detectMarkers(img2, dictionary, corners2, ids2);

    std::vector<cv::Vec3d> rvecs1, tvecs1;
    std::vector<cv::Vec3d> rvecs2, tvecs2;

    // Camera 1 processing
    cv::aruco::drawDetectedMarkers(img1, corners1, ids1);
    cv::aruco::estimatePoseSingleMarkers(corners1, marker_length_m,
                                                 camera_matrix1, dist_coeffs1, rvecs1, tvecs1);

    std::cout << "Translation: " << tvecs1[0]
              << "\nRotation: " << rvecs1[0]
              << std::endl;

    cv::aruco::drawAxis(img1, camera_matrix1, dist_coeffs1,
                            rvecs1[0], tvecs1[0], 0.1);

    imshow("Pose estimation", img1);
    cv::waitKey(0);


    // Camera 2 processing
    cv::aruco::drawDetectedMarkers(img2, corners2, ids2);
    cv::aruco::estimatePoseSingleMarkers(corners2, marker_length_m,
                                             camera_matrix2, dist_coeffs2, rvecs2, tvecs2);

    std::cout << "Translation: " << tvecs2[0]
              << "\nRotation: " << rvecs2[0]
              << std::endl;

    cv::aruco::drawAxis(img2, camera_matrix2, dist_coeffs2,
                            rvecs2[0], tvecs2[0], 0.1);

    imshow("Pose estimation", img2);
    cv::waitKey(0);


    // Compute transformation matrix 2 to 1
    Mat R1, R2;
    Rodrigues(rvecs1[0], R1);
    Rodrigues(rvecs2[0], R2);

    Mat R_1to2;
    Mat t_1to2;

    cv::Mat T1(3, 1, CV_32FC1);
    T1.at<float>(0, 0) = tvecs1[0][0];
    T1.at<float>(1, 0) = tvecs1[0][1];
    T1.at<float>(2, 0) = tvecs1[0][2];

    cv::Mat T2(3, 1, CV_32FC1);
    T2.at<float>(0, 0) = tvecs2[0][0];
    T2.at<float>(1, 0) = tvecs2[0][1];
    T2.at<float>(2, 0) = tvecs2[0][2];

    cout << tvecs1[0] << "\n";
    cout << T1 << "\n";

    cout << tvecs2[0] << "\n";
    cout << T2 << "\n";

    computeC2MC1(R1, T1, R2, T2, R_1to2, t_1to2);

    Mat rvec_1to2;
    Rodrigues(R_1to2, rvec_1to2);

    cout << "Result: " << endl;
    std::cout << R_1to2 << std::endl;
    std::cout << t_1to2 << std::endl;
    std::cout << rvec_1to2 << std::endl;

    return 0;
}