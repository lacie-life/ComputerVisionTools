#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>       

int main(int argc, char * argv[])
{
    rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);

    rs2::rates_printer printer;
    std::vector<rs2::pipeline> pipelines;
    std::map<std::string, rs2::colorizer> colorizers;
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
        colorizers[serial] = rs2::colorizer();
    }

    
    std::map<int, rs2::frame> render_frames;

    cv::Mat image;

    int left = 0;
    int right = 0;
    char left_path[100];
    char right_path[100];

    while (1) 
    {
        std::vector<rs2::frame> new_frames;
        std::vector<cv::Mat> new_images;
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

       

        char c = (char)cv::waitKey(25);
        if(c == 27){
            sprintf(left_path, "/home/jun/Github/ComputerVisionTools/realsense_capture/images/left/0-%d.jpg", left);
            sprintf(right_path, "/home/jun/Github/ComputerVisionTools/realsense_capture/images/right/1-%d.jpg", right);

            std::cout << left_path << std::endl;
            std::cout << right_path << std::endl;
    
            if (!new_images.empty()){
                cv::imwrite(left_path, new_images.at(0));
                cv::imwrite(right_path, new_images.at(1));
                left++;
                right++;
            }
        }
    }

    return 0;
}