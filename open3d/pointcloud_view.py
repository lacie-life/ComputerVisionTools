import open3d as o3d

# Load the point cloud from a PCD file
pcd = o3d.io.read_point_cloud("/mnt/Data-2/Datasets/A9_dataset_all/A9_dataset/training/velodyne/south/south1/1646667326_151561023_s110_lidar_ouster_south.pcd")

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the point cloud to the visualization
vis.add_geometry(pcd)

# Set the camera view point
vis.get_view_control().set_front([0, 0, -1])
vis.get_view_control().set_lookat([0, 0, 0])
vis.get_view_control().set_up([0, -1, 0])
vis.get_view_control().set_zoom(0.5)

opt = vis.get_render_option()
opt.point_size = 0.01 

# Run the visualization
vis.run()
vis.destroy_window()