#include <iostream>
#include <thread>
#include <chrono>
#include <math.h>
#include <vector>
#include <mutex>

#include <pcl/io/pcd_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std::chrono;
using namespace std;

Eigen::MatrixXd traj(10000,3);
int idx_list[20] = {150,300};
int final_end = 300;

shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud){
    // Open 3D viewer and add point cloud
    shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("PCL ZED 3D Viewer"));
    viewer->setBackgroundColor(0.12, 0.12, 0.12);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
}

inline float convertColor(float colorIn){
    uint32_t color_uint = *(uint32_t *) & colorIn;
    unsigned char *color_uchar = (unsigned char *) &color_uint;
    color_uint = ((uint32_t) color_uchar[0] << 16 | (uint32_t) color_uchar[1] << 8 | (uint32_t) color_uchar[2]);
    return *reinterpret_cast<float *> (&color_uint);
}

int main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr p_pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    p_pcl_point_cloud->points.resize((640, 480));

    shared_ptr<pcl::visualization::PCLVisualizer> viewer = createRGBVisualizer(p_pcl_point_cloud);

    // Set Viewer initial position
    viewer->setCameraPosition(0, 0, 5,    0, 0, 1,   0, 1, 0);
    viewer->setCameraClipDistances(0.1,1000);

    // Loop until viewer catches the stop signal
    while (!viewer->wasStopped()) {

//        //Lock to use the point cloud
//        mutex_input.lock();
//        float *p_data_cloud = data_cloud.getPtr<float>();
//        int index = 0;
//
//        // Check and adjust points for PCL format
//        for (auto &it : p_pcl_point_cloud->points) {
//            float X = p_data_cloud[index];
//            if (!isValidMeasure(X)) // Checking if it's a valid point
//                it.x = it.y = it.z = it.rgb = 0;
//            else {
//                it.x = X;
//                it.y = p_data_cloud[index + 1];
//                it.z = p_data_cloud[index + 2];
//                it.rgb = convertColor(p_data_cloud[index + 3]); // Convert a 32bits float into a pcl .rgb format
//            }
//            index += 4;
//        }
//
//        // Unlock data and update Point cloud
//        mutex_input.unlock();
//        viewer->updatePointCloud(p_pcl_point_cloud);
        viewer->spinOnce(10);
    }

    // Close the viewer
    viewer->close();

    return 0;
}
