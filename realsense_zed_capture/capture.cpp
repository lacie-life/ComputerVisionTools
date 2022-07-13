// Realsense includes 
#include <librealsense2/rs.hpp>
 // ZED includes
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>   

cv::Mat slMat2cvMat(sl::Mat& input);
#ifdef HAVE_CUDA
cv::cuda::GpuMat slMat2cvMatGPU(sl::Mat& input);
#endif // HAVE_CUDA

int main(int argc, char * argv[])
{
    // Realsense config
    rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);

    rs2::rates_printer printer;
    std::vector<rs2::pipeline> pipelines;
    rs2::context ctx;

    // Capture serial numbers before opening streaming
    std::vector<std::string> serials;
    for (auto&& dev : ctx.query_devices())
    {
        serials.push_back(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    }

    // Start a streaming pipe per each connected device
    for (auto&& serial : serials)
    {
        rs2::pipeline pipe(ctx);
        rs2::config cfg;
        cfg.enable_device(serial);
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        pipe.start(cfg);
        pipelines.emplace_back(pipe);
    }

    // ZED camera config
    // Create a ZED camera object
    sl::Camera zed;

    // Set configuration parameters
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD1080;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.coordinate_units = sl::UNIT::METER;
        
    // Open the camera
    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        printf("%s\n", toString(err).c_str());
        zed.close();
        return 1; // Quit if an error occurred
    }

    sl::CalibrationParameters calibration_params = zed.getCameraInformation().camera_configuration.calibration_parameters;

    std::cout << "Left camera calibration: " << std::endl;
    std::cout << "Fx: " << calibration_params.left_cam.fx << "\n";
    std::cout << "Fy: " << calibration_params.left_cam.fy << "\n";
    std::cout << "Cx: " << calibration_params.left_cam.cx << "\n";
    std::cout << "Cx: " << calibration_params.left_cam.cy << "\n";

    std::cout << "Distortion: " << calibration_params.left_cam.disto[0] << " "
                                << calibration_params.left_cam.disto[1] << " "
                                << calibration_params.left_cam.disto[2] << " "
                                << calibration_params.left_cam.disto[3] << " "
                                << calibration_params.left_cam.disto[4] << "\n";

    // Set runtime parameters after opening the camera
    sl::RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = sl::SENSING_MODE::STANDARD;

    // Prepare new image size to retrieve half-resolution images
    sl::Resolution image_size = zed.getCameraInformation().camera_resolution;
    int new_width = image_size.width / 2;
    int new_height = image_size.height / 2;

    sl::Mat image_zed(new_width, new_height, sl::MAT_TYPE::U8_C4);

    sl::Resolution new_image_size(new_width, new_height);

    int left = 0;
    int right = 0;
    int center = 0;
    char left_path[100];
    char right_path[100];
    char center_path[100];

    char key = ' ';
    while (key != 'q') 
    {
        std::vector<cv::Mat> new_images;

        std::vector<rs2::frame> new_frames;
        for (int i = 0; i < 2; i++)
        {

            rs2::frameset fs = pipelines.at(i).wait_for_frames();
            rs2::frame colorImage = fs.get_color_frame();
            auto name = rs2::sensor_from_frame(colorImage)->get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
            cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)colorImage.get_data(), cv::Mat::AUTO_STEP);
            // cv::cvtColor(color, color, cv::COLOR_BGR2RGB);
            cv::imshow(name, color);

            if (color.empty()) {
                std::cerr << "ERROR! blank frame grabbed\n";
                break;
            }

            new_images.emplace_back(color);
        }

        if(zed.grab(runtime_parameters) == sl::ERROR_CODE::SUCCESS)
        {
            cv::Mat image_ocv;

            // Retrieve the left image, depth image in half-resolution
            zed.retrieveImage(image_zed, sl::VIEW::LEFT, sl::MEM::CPU, new_image_size);
            cv::Mat img_rgba = slMat2cvMat(image_zed);
            cv::cvtColor(img_rgba, image_ocv, cv::COLOR_BGRA2BGR);

            cv::imshow("ZED 2", image_ocv);

            if(!new_images.empty())
            {
                new_images.emplace_back(image_ocv);
            }
        }

        char c = (char)cv::waitKey(25);
        if(c == 27){
            sprintf(right_path, "/home/jun/Github/ComputerVisionTools/realsense_zed_capture/images/right/0-%d.jpg", right);
            sprintf(left_path, "/home/jun/Github/ComputerVisionTools/realsense_zed_capture/images/left/1-%d.jpg", left);
            sprintf(center_path, "/home/jun/Github/ComputerVisionTools/realsense_zed_capture/images/center/2-%d.jpg", center);

            std::cout << left_path << std::endl;
            std::cout << right_path << std::endl;
            std::cout << center_path << std::endl;
    
            if (!new_images.empty()){
                cv::imwrite(right_path, new_images.at(0));
                cv::imwrite(left_path, new_images.at(1));
                cv::imwrite(center_path, new_images.at(2));
                left++;
                right++;
                center++;
            }
        }
    }
    
    zed.close();
    return 0;

    return 0;
}

// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), 
                   input.getWidth(), 
                   getOCVtype(input.getDataType()), 
                   input.getPtr<sl::uchar1>(sl::MEM::CPU), 
                   input.getStepBytes(sl::MEM::CPU));
}

#ifdef HAVE_CUDA
/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::cuda::GpuMat slMat2cvMatGPU(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::cuda::GpuMat(input.getHeight(), 
                            input.getWidth(), 
                            getOCVtype(input.getDataType()), 
                            input.getPtr<sl::uchar1>(sl::MEM::GPU), 
                            input.getStepBytes(sl::MEM::GPU));
}
#endif
