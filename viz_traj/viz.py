import argparse
import os

import numpy as np
import open3d as o3d


def parse_pose(filepath):
    proj_matrix_list = []
    with open(filepath, 'r') as fin:
        for line in fin:
            _inv_proj_matrix = list(map(float, line.strip().split()))
            _inv_proj_matrix = np.array(_inv_proj_matrix).reshape(3, 4)
            inv_proj_matrix = np.zeros((4, 4), dtype=np.float32)
            inv_proj_matrix[:3, :4] = _inv_proj_matrix
            inv_proj_matrix[3, 3] = 1.0

            # proj_matrix = np.zeros((3, 4), dtype=np.float32)
            # proj_matrix[:3, :3] = inv_proj_matrix[:3, :3].transpose()
            # proj_matrix[:, 4] = - inv_proj_matrix[:3, :3].dot(inv_proj_matrix[:, 4])

            proj_matrix_list.append(inv_proj_matrix)

    return proj_matrix_list



CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])


def build_camera_model(scale=0.1):
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (1.0, 0, 0)
    camera_actor.paint_uniform_color(color)

    return camera_actor


def viz_camera_trajectory(proj_matrix_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for proj_matrix in proj_matrix_list:
        cam = build_camera_model()
        cam.transform(proj_matrix)
        vis.add_geometry(cam)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pose_file', type=str)
    args = parser.parse_args()

    pose = parse_pose(args.pose_file)
    viz_camera_trajectory(pose)
