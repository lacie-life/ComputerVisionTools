#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <thread>
#include <string>
#include <stdlib.h>
#include <unistd.h>

#include <fstream>

#include "SparkFun_u-blox_GNSS_Arduino_Library.h"

// Create gstreamer capture command
std::string gstreamer_pipeline (int sensor_id = 0, 
                                int capture_width = 4032, 
                                int capture_height = 3040, 
                                int display_width = 4032, 
                                int display_height = 3040, 
                                int framerate = 30, 
                                int flip_method = 2) {

    return "nvarguscamerasrc sensor_id= " + 
            std::to_string(sensor_id) + 
            " ! video/x-raw(memory:NVMM), width=(int)" + 
            std::to_string(capture_width) + 
            ", height=(int)" +
           std::to_string(capture_height) + 
           ", format=(string)NV12, framerate=(fraction)" + 
           std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + 
           std::to_string(flip_method) + 
           " ! video/x-raw, width=(int)" + 
           std::to_string(display_width) + 
           ", height=(int)" +
           std::to_string(display_height) + 
           ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// Take picture
void take_picture(SFE_UBLOX_GNSS gps, int num){

    std::string pipeline = gstreamer_pipeline();
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if(!cap.isOpened()) {
        std::cout<<"Failed to open camera."<<std::endl;
        exit(-1);
    }

    cv::Mat img;

    bool result = true;

    while(result)
    {
    	if (cap.read(img)) {
            std::string name = std::to_string(num) + ".jpg";
            cv::imwrite(name, img);
            result = false;

            printf("Picture timeline: \n");
            printf ("%02d/%02d/%02d %02d:%02d:%02d %d:%d\n", gps.getDay(), gps.getMonth(), gps.getYear(), 
                                                           gps.getHour(), gps.getMinute(), gps.getSecond(), 
                                                           gps.getMillisecond(), gps.getNanosecond());
            printf("Latitude                : %2.8f (deg)\n", gps.getLatitude() * 1e-7);
            printf("Longitude               : %2.8f (deg)\n", gps.getLongitude() * 1e-7);
	    }
    }

}

// GPS object
SFE_UBLOX_GNSS myGPS;

int main(int argc, char** argv)
{

    // GPS Setting
    if(argc == 1) {
        printf("\nublox_f9p_test <ublox_com> <pseudo_com> (ublox_f9p_test '/dev/ttyACM0')"); 
        return 0;
    } else if (argc == 2) {
        for(int counter=0;counter<argc;counter++) 
            printf("\nargv[%d]: %s",counter,argv[counter]);        
    } else if(argc >= 3) {
        printf ("\nMore number of arguments...");
        return 0;
    } 

    Stream seriComm(argv[1]);
    seriComm.begin(38400);
    if (!seriComm.isConnected()) {
        printf ("Ublox is not connected. Please connect ublox GNSS module and try again...\n");
        return 0;
    }

    myGPS.begin(seriComm);
    myGPS.setNavigationFrequency(8); //Set output to 8 times a second
    myGPS.saveConfiguration(); //Save the current settings to flash and BBR

    int count = 0;

    std::string pipeline = gstreamer_pipeline();
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if(!cap.isOpened()) {
        std::cout<<"Failed to open camera."<<std::endl;
        exit(-1);
    }

    cv::Mat img;

    std::ofstream gpsFile;
    gpsFile.open("gps.txt");
    
    printf ("\n--------------------------------------------------------\n");

    while(true) {

        if (myGPS.getPVT()) {
                printf ("%02d/%02d/%02d %02d:%02d:%02d %d:%d\n", myGPS.getDay(), myGPS.getMonth(), myGPS.getYear(), 
                                                            myGPS.getHour(), myGPS.getMinute(), myGPS.getSecond(), 
                                                            myGPS.getMillisecond(), myGPS.getNanosecond());
                printf("Latitude                : %2.8f (deg)\n", myGPS.getLatitude() * 1e-7);
                printf("Longitude               : %2.8f (deg)\n", myGPS.getLongitude() * 1e-7);
                printf("Altitude                : %d (mm)\n", myGPS.getAltitude());
                printf("Altitude MSL            : %d (mm)\n", myGPS.getAltitudeMSL());	

                gpsFile << count << " ";
                gpsFile << myGPS.getDay() << "/" 
                        << myGPS.getMonth() << "/" 
                        << myGPS.getYear() << " " 
                        << myGPS.getHour() << ":" 
                        << myGPS.getMinute() << ":"
                        << myGPS.getSecond() << ":"
                        << myGPS.getMillisecond() << ":"
                        << myGPS.getNanosecond() << "\n";
                gpsFile << "Latitude    : " << myGPS.getLatitude() * 1e-7 << "\n";
                gpsFile << "Longitude   : " << myGPS.getLongitude() * 1e-7 << "\n";
                gpsFile << "Altitude    : " << myGPS.getAltitude() << "\n";
                gpsFile << "Altitude MSL: " << myGPS.getAltitudeMSL() << "\n \n";
         }
        if (!cap.read(img)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }

        std::string name = std::to_string(count) + ".jpg";
        cv::imwrite(name, img);
        //cv::imshow("Test", img);
        count = count + 1;

        int keycode = cv::waitKey(30) & 0xff ; 
        if (keycode == 27) break ;

        sleep(1);

    }
    cap.release();
    cv::destroyAllWindows() ;

    return 1;
}
