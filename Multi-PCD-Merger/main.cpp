#include <string>
#include <iostream>
#include <filesystem>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

namespace fs = std::filesystem;

int main() {
    std::string path = "/home/lacie/Loam_livox_loop/aft_mapp/";
    int i = 0;

    // The PCD files
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>); // Create a point cloud （ The pointer ）
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_prt(new pcl::PointCloud<pcl::PointXYZ>); // Create a point cloud （ The pointer ）
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>); // Create a point cloud （ The pointer ）


    for (const auto &entry: fs::directory_iterator(path)){
        std::cout << entry.path() << std::endl;
        pcl::io::loadPCDFile(entry.path(), *cloud_prt);

        Eigen::Vector4f p = cloud_prt->sensor_origin_.matrix();
        Eigen::Quaternionf q;
        q = cloud_prt->sensor_orientation_.matrix();
        Eigen::Vector3f eulerAngle = q.matrix().eulerAngles(2, 1, 0);
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translation() << p[0], p[1], p[2];
        transform.rotate(q);
        std::cout << transform.matrix() << std::endl;
        pcl::transformPointCloud(*cloud_prt, *transformed_cloud, transform);
        *cloud += *transformed_cloud;
        cloud_prt->clear();
        transformed_cloud->clear();

        usleep(100);

        i++;
    }
    std::cout << i << std::endl;
    // Save the output file
    pcl::io::savePCDFileASCII("output.pcd", *cloud);

    return 0;
}
