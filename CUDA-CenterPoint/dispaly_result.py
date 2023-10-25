import open3d 
import numpy as np
import os
import time

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d

# 3D界面初始化
vis = open3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().point_size = 1.0
vis.get_render_option().background_color = np.zeros(3)
        
pc_path = "/data/chenghao/mycode/CUDA-CenterPoint-WZ/data/wz_data/"
det_path = "/data/chenghao/mycode/CUDA-CenterPoint-WZ/data/prediction/"
fileList = os.listdir(pc_path)
fileList.sort()
# 遍历文件夹中的文件
for filename in fileList:
    base_name = filename[:-3]
    file_path = os.path.join(pc_path, filename)
    if os.path.isfile(file_path):
        # 处理文件
        # print(base_name)
        pc_file = os.path.join(pc_path, filename)
        det_file = os.path.join(det_path, base_name+"txt")

        # 加载点云数据
        points = np.fromfile(pc_file, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        # print(points)
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(points)

        vis.add_geometry(point_cloud)

        det = np.loadtxt(det_file,dtype=np.float32)
        print(det.shape)
        for i in range(det.shape[0]):
            line_set, box3d = translate_boxes_to_open3d_instance(det[i,:])
            line_set.paint_uniform_color((0, 1, 0))
            vis.add_geometry(line_set)
    
        vis.poll_events()
        vis.update_renderer()
        vis.clear_geometries()
        time.sleep(0.1)
        # vis.run()