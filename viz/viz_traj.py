import argparse
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import pykitti


def get_inv_proj_matrix(proj_matrix):
    inv_proj_matrix = np.zeros((3, 4), dtype=np.float32)
    inv_proj_matrix[:3, :3] = proj_matrix[:3, :3].transpose()
    inv_proj_matrix[:, 4] = - proj_matrix[:3, :3].dot(proj_matrix[:, 4])
    return inv_proj_matrix


def parse_pose(filepath):
    proj_matrix_list = []
    with open(filepath, 'r') as fin:
        for line in fin:
            _inv_proj_matrix = list(map(float, line.strip().split()))
            _inv_proj_matrix = np.array(_inv_proj_matrix).reshape(3, 4)
            inv_proj_matrix = np.zeros((4, 4), dtype=np.float32)
            inv_proj_matrix[:3, :4] = _inv_proj_matrix
            inv_proj_matrix[3, 3] = 1.0

            proj_matrix_list.append(inv_proj_matrix)

    return proj_matrix_list


def parse_camera(filepath):
    with open(filepath, 'r') as fin:
        data = fin.readlines()
    intrinsic = data[2].split(maxsplit=1)[1]
    intrinsic = list(map(float, intrinsic.split()))
    intrinsic = np.array(intrinsic).reshape(3, 4)[:3, :3]
    return intrinsic


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
    img = cv2.imread(dataset.cam2_files[0])
    height, width = img.shape[:2]
    return width, height


def vis_trajectory(dataset):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    intrinsic = dataset.calib.K_cam2
    width, height = get_image_size(dataset)
    for idx, pose in enumerate(dataset.poses):
        cam = build_camera_model(intrinsic[0,0], width, height)
        cam.transform(pose)
        vis.add_geometry(cam)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_odometry_path', type=str, default='/home/ruanzhiwei/workspace/dataset/dataset')
    parser.add_argument('--sequence', type=str, default='00')
    args = parser.parse_args()

    dataset = pykitti.odometry(args.kitti_odometry_path, args.sequence)
    vis_trajectory(dataset)
