#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());

    return  cv::norm(I, shouldBeIdentity) < 1e-6;

}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
{

    assert(isRotationMatrix(R));

    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);

}

int main()
{
    cv::Mat center2left = cv::Mat_<double>(4, 4);
    center2left.at<double>(0, 0)=  0.98543566;
    center2left.at<double>(0, 1)=  0.042277779;
    center2left.at<double>(0, 2)=  0.16470966;
    center2left.at<double>(0, 3)=  -0.39072222;

    center2left.at<double>(1, 0)=  -0.078468107;
    center2left.at<double>(1, 1)=  0.97236675;
    center2left.at<double>(1, 2)=  0.21987633;
    center2left.at<double>(1, 3)=  0.27189398;

    center2left.at<double>(2, 0)=  -0.15086232;
    center2left.at<double>(2, 1)=  -0.2295984;
    center2left.at<double>(2, 2)=  0.96152234;
    center2left.at<double>(2, 3)=  -0.62192124;

    center2left.at<double>(3, 0)=  0.0;
    center2left.at<double>(3, 1)=  0.0;
    center2left.at<double>(3, 2)=  0.0;
    center2left.at<double>(3, 3)=  1.0;

    cv::Mat right2center = cv::Mat_<double>(4, 4);
    right2center.at<double>(0, 0)=  0.99351341;
    right2center.at<double>(0, 1)=  0.11242391;
    right2center.at<double>(0, 2)=  0.017086796;
    right2center.at<double>(0, 3)=  0.36860722;

    right2center.at<double>(1, 0)=  -0.1058159;
    right2center.at<double>(1, 1)=  0.96902907;
    right2center.at<double>(1, 2)=  -0.22312708;
    right2center.at<double>(1, 3)=  0.46355382;

    right2center.at<double>(2, 0)=  -0.041642416;
    right2center.at<double>(2, 1)=  0.2198717;
    right2center.at<double>(2, 2)=  0.97463959;
    right2center.at<double>(2, 3)=  -0.26471555;

    right2center.at<double>(3, 0)=  0.0;
    right2center.at<double>(3, 1)=  0.0;
    right2center.at<double>(3, 2)=  0.0;
    right2center.at<double>(3, 3)=  1.0;

    cv::Mat right2left = cv::Mat_<double>(4, 4);
    right2left = right2center * center2left;

    cv::Mat T = cv::Mat_<double>(3, 1);
    cv::Mat R_matrix = cv::Mat_<double>(3, 3);

    T.at<double>(0, 0) = right2left.at<double>(0, 3);
    T.at<double>(1, 0) = right2left.at<double>(1, 3);
    T.at<double>(2, 0) = right2left.at<double>(2, 3);

    R_matrix.at<double>(0, 0)=  right2left.at<double>(0, 0);
    R_matrix.at<double>(0, 1)=  right2left.at<double>(0, 1);
    R_matrix.at<double>(0, 2)=  right2left.at<double>(0, 2);
    R_matrix.at<double>(1, 0)=  right2left.at<double>(1, 0);
    R_matrix.at<double>(1, 1)=  right2left.at<double>(1, 1);
    R_matrix.at<double>(1, 2)=  right2left.at<double>(1, 2);
    R_matrix.at<double>(2, 0)=  right2left.at<double>(2, 0);
    R_matrix.at<double>(2, 1)=  right2left.at<double>(2, 1);
    R_matrix.at<double>(2, 2)=  right2left.at<double>(2, 2);

    cv::Vec3f RPY = rotationMatrixToEulerAngles(R_matrix);

    std::cout << "Result: " << std::endl;
    std::cout << right2left << std::endl;
    std::cout << T << std::endl;
    std::cout << R_matrix << std::endl;
    std::cout << RPY << std::endl;

    return 0;
}