import argparse
import os
import sys

import numpy as np
import open3d as o3d
from PIL import Image
import pykitti


def build_camera_model(f, w, h, scale=0.001):
    CAM_POINTS = np.array([
            [ 0,   0,   0],
            [-w/2,  -h/2, f],
            [ w/2,  -h/2, f],
            [ w/2,   h/2, f],
            [-w/2,   h/2, f],
            ])

    CAM_LINES = np.array([
        [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4]])

    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (1.0, 0, 0)
    camera_actor.paint_uniform_color(color)

    return camera_actor


def get_image_size(dataset):
    img = Image.open(dataset.cam2_files[0])
    return img.size


def _depth2cloud(intrinsic, depth_img_path, rgb_img_path):
    depth = np.array(Image.open(depth_img_path)).astype(np.float32) / 256
    mask = depth > 0

    X, Y, Z = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]), 1)
    X, Y, Z = X[mask].reshape(1, -1), Y[mask].reshape(1, -1), Z[mask].reshape(1, -1)
    xyz = np.vstack([X, Y, Z]).astype(np.float32)
    valid_depth = depth[mask].reshape(-1)

    xyz -= np.array([intrinsic[0, 2], intrinsic[1, 2], 0]).reshape(3, -1)
    xyz *= valid_depth
    xyz /= np.array([intrinsic[0, 0], intrinsic[1, 1], 1]).reshape(3, -1)

    colors = np.array(Image.open(rgb_img_path))
    colors = colors[mask].reshape(-1, 3)

    return xyz, colors


def depth2cloud(dataset, depth_img_dir, num_images):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    intrinsic = dataset.calib.K_cam2
    width, height = get_image_size(dataset)
    depth_img_list = sorted(os.listdir(depth_img_dir))

    T2 = np.eye(4)
    T2[0, 3] = dataset.calib.P_rect_20[0, 3] / dataset.calib.P_rect_20[0, 0]
    for depth_img_name in depth_img_list[:num_images]:
        depth_img_path = os.path.join(depth_img_dir, depth_img_name)
        idx = int(os.path.basename(depth_img_path).split('.')[0])
        extrinsic = dataset.poses[idx]
        extrinsic = extrinsic.dot(np.linalg.inv(T2))
        rgb_img_path = dataset.cam2_files[idx]
        points, colors = _depth2cloud(intrinsic, depth_img_path, rgb_img_path)
        points = points.transpose().astype(np.float64)
        colors = colors.astype(np.float64) / 255
        cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        cloud.transform(extrinsic)
        vis.add_geometry(cloud)

        cam = build_camera_model(intrinsic[0,0], width, height)
        cam.transform(extrinsic)
        vis.add_geometry(cam)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_odometry_path', type=str,
            default='/home/ruanzhiwei/workspace/dataset/dataset')
    # parser.add_argument('--depth_image_path', type=str,
    #         default='/home/ruanzhiwei/workspace/dataset/depth/train/2011_10_03_drive_0027_sync/proj_depth/groundtruth/image_02')
    parser.add_argument('--depth_image_path', type=str,
            default='/home/ruanzhiwei/workspace/dataset/depth/train/2011_10_03_drive_0027_sync/proj_depth/velodyne_raw/image_02')
    parser.add_argument('--sequence', type=str, default='00')
    parser.add_argument('--num_images', type=int, default=50)
    args = parser.parse_args()

    dataset = pykitti.odometry(args.kitti_odometry_path, args.sequence)
    depth2cloud(dataset, args.depth_image_path, args.num_images)
