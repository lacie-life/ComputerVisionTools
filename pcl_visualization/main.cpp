#include <iostream>
#include <thread>
#include <chrono>
#include <math.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vector>

using namespace std::chrono;
Eigen::MatrixXd traj(10000,3);
int idx_list[20] = {150,300};
int final_end = 300;

//generate simple camera position
bool gen_traj_cmd()
{
    int i = 1;
    traj.row(0)<<0,1,15;//initialize

    while(i<final_end){
        //forward
        if(i<idx_list[0]){
            traj.row(i)<< traj(i-1,0), traj(i-1,1), traj(i-1,2)-0.08;
        }
        //backward
        else if(i<idx_list[1]){
            traj.row(i)<< traj(i-1,0), traj(i-1,1), traj(i-1,2)+0.08;
        }
        i++;
    }
}

pcl::visualization::PCLVisualizer::Ptr mapping_vis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

int main (int argc, char** argv)
{

    //initialization
    int idx = 0;
    Eigen::Vector3d pos, focal, up_vector;
    std::vector<pcl::visualization::Camera> Cameras;

    std::string file_name (argv[1]);
    //read map
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map (new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (file_name, *cloud_map) == -1) //* load the file
    {
        PCL_ERROR ("Couldn’t read pcd file \n");
        return (-1);
    }
    std::cout << "Loaded "
    << cloud_map->width * cloud_map->height
            << " data points from loaded map with the following fields: "
    << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = mapping_vis(cloud_map);

    gen_traj_cmd();
    focal << 0,0,0;//initialize focal point.

    while (!viewer->wasStopped ())
    {
        //get camera position, focal point, and up vector
        viewer->getCameras(Cameras);
        pos << Cameras[0].pos[0],Cameras[0].pos[1],Cameras[0].pos[2];
        focal << Cameras[0].focal[0], Cameras[0].focal[1], Cameras[0].focal[2];
        up_vector << Cameras[0].view[0],Cameras[0].view[1],Cameras[0].view[2];

        //set camera focal point
        focal<<pos(0), pos(1), pos(2)-10;

        // update camera position, focal point, and up vector
        if(idx<final_end){
            viewer->setCameraPosition(traj(idx,0),traj(idx,1),traj(idx,2),focal(0),focal(1),focal(2),up_vector(0),up_vector(1),up_vector(2),0);
        }else{
            viewer->spinOnce(0); //trajectory end. get keyboard and mouse callback.
        }

        milliseconds dura(5);//loop for 5 miliseconds
        std::this_thread::sleep_for(dura);

        idx++;
    }
}
