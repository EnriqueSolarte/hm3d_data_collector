import dvf_map
import hydra
import numpy as np
from imageio import imwrite
import matplotlib.pyplot as plt
import logging
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.io_utils import get_abs_path, create_directory
import os
from imageio.v3 import imread, imwrite
from matplotlib import pyplot as plt
import json 
import pandas as pd
from geometry_perception_utils.spherical_utils import uv2xyz
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous, eulerAnglesToRotationMatrix
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.vispy_utils import plot_list_pcl, plot_color_plc
from pyquaternion import Quaternion
from tqdm import trange
# methods in this experiment
from dvf_map.experiments.pre_process_hm.utils import read_positions, project_pp_depth, get_cam_pose
from dvf_map.dense_voxel_grid.voxel_grid_2d import VoxelGrid2D
from dvf_map.dense_voxel_grid.voxel_grid_3d import VoxelGrid3D

        
        
@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    
    positions = read_positions(cfg.hm_data.gt_position_fn)
    semantics = pd.read_csv(cfg.hm_data.semantic_fn)
    
    voxel = VoxelGrid2D(cfg.voxel_grid_2d)
    # voxel = VoxelGrid3D(cfg.voxel_grid_3d)

    rgb_dir = f"{cfg.hm_data.rgb_dir}"
    depth_dir = f"{cfg.hm_data.depth_dir}"
    semantics_dir = f"{cfg.hm_data.semantic_dir}"
    
    pcl = []
    for idx in trange(len(positions)):
        rgb = imread(f"{rgb_dir}/{idx}.jpg")
        semantics_img = np.load(f"{semantics_dir}/{idx}.npy")        
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
                
        xyz, m = project_pp_depth(depth_map, fov=cfg.hm_data.cfg.habitat.hfov)
        color_xyz = get_color_array(rgb)[:, m]/255
        semantic_xyz = get_color_array(semantics_img)[0][m]
        cam_pose = get_cam_pose(positions[idx])
        
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz)
        xyz_wc_vx, vx_idx, xyz_idx, all_xyz_idx = voxel.project_xyz(xyz_wc)
        # vx --> xyz
        # print(xyz_idx.shape, xyz_wc_vx.shape, xyz_idx.max())
        # xyz --> vx
        # print(all_xyz_idx.shape, xyz_wc.shape, all_xyz_idx.max())
        
        local_xyz = np.vstack([xyz_wc_vx, color_xyz[:, xyz_idx]])
        idx = np.linspace(0, local_xyz.shape[1]-1, 100, dtype=np.int32)
        np.random.shuffle(idx)
        pcl.append(local_xyz[:, idx])
        
        # plot_color_plc(xyz_wc_vx.T, color_xyz[:, xyz_idx].T)
        # # building vxl map from indexes
        # vxl_centers = voxel.get_centroids_by_idx(vx_idx)
        # plot_color_plc(vxl_centers.T, color_xyz[:, xyz_idx].T)
        
    pcl = np.hstack(pcl)
    plot_color_plc(pcl[:3, :].T, pcl[3:, :].T)
    
        
if __name__ == "__main__":
    main()