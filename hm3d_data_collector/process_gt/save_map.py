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
from dvf_map.experiments.pre_process_hm.utils import project_pp_depth, get_cam_pose
from dvf_map.dense_voxel_grid.voxel_grid_2d import VoxelGrid2D
from dvf_map.dense_voxel_grid.voxel_grid_3d import VoxelGrid3D
from dvf_map.ray_tracer.ray_tracer import RaysTracer
from geometry_perception_utils.image_utils import get_color_array, get_color_list
from geometry_perception_utils.io_utils import save_obj, load_obj
from dvf_map.experiments.pre_process_hm.utils import (
    read_positions, project_pp_depth, get_cam_pose, 
    project_xyz_to_lidar, mask_semantics, get_bearings)


class SemanticMapRef:
    xyz_sem_2d: dict
    voxel_2d: list
    xyz_sem_3d: dict
    voxel_3d: list
    xyz_lidar: dict
    id: dict


def project_xyz_semantics(sem, xyz, xyz_mask, list_id, dict_xyz, fov):
    id_sem = get_color_array(sem)[0]
    
    # masking valid points by passed mask
    id_sem = id_sem[xyz_mask]

    # get only xyz for passed semantic id list
    # append xyz by id
    [dict_xyz[id].append(xyz[:, id_sem == id]) for id in list_id if id != 0 and np.sum(id_sem == id) > 0]
    

    
@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    
    positions = read_positions(cfg.hm_data.gt_position_fn)
    semantics = pd.read_csv(cfg.hm_data.semantic_fn)
    
    depth_dir = f"{cfg.hm_data.depth_dir}"
    semantics_dir = f"{cfg.hm_data.semantic_dir}"
    
    sem_colors = get_color_list(number_of_colors=semantics.shape[0])
    sem_colors = sem_colors[:, np.random.permutation(semantics.shape[0])]
    sem_colors[:, 0] = np.array([0, 0, 0])
        
    # Ray casting projector fro mimic the lidar
    ray_tracer = RaysTracer(cfg.hm_data.lidar.ray_tracer)
    
    # Voxel map for the data
    voxel_2d_map = VoxelGrid2D(cfg.voxel_grid_2d)
    voxel_3d_map = VoxelGrid3D(cfg.voxel_grid_3d)
    
    scene_lidar_xyz = []
    dict_bev_maps = {id: [] for id in semantics['id']}

    list_id = list(semantics.id)
    
    for idx in trange(len(positions)):
    # for idx in trange(10):
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
        semantics_map = np.load(f"{semantics_dir}/{idx}.npy")
    
        # xyz in CC 
        xyz_cc, xyz_mask = project_pp_depth(depth_map, fov=cfg.hm_data.cfg.habitat.hfov)
        cam_pose = get_cam_pose(positions[idx])
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_cc)

        ######### xyz in lidar ###############
        xyz_lidar_cc, ret = project_xyz_to_lidar(xyz_cc, ray_tracer, cfg.hm_data.lidar)
        if ret:
            xyz_lidar_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_lidar_cc)
            scene_lidar_xyz.append(xyz_lidar_wc)
        
        ######## Process semantic data ############
        project_xyz_semantics(semantics_map, xyz_wc, xyz_mask, list_id, dict_bev_maps, fov=cfg.hm_data.cfg.habitat.hfov)
        
    # Aggregate semantic from multi-view
    dict_bev_maps = {id: np.hstack(dict_bev_maps[id]) for id in dict_bev_maps.keys() if id != 0 and dict_bev_maps[id].__len__() > 0}
    
    # 2D and 3D Semantic voxel maps 
    xyz_sem_2d = {id: voxel_2d_map.project_xyz(xyz)[0] for id, xyz in dict_bev_maps.items() if id != 0}
    xyz_sem_3d = {id: voxel_3d_map.project_xyz(xyz)[0] for id, xyz in dict_bev_maps.items() if id != 0}

    xyz_lidar = voxel_2d_map.project_xyz(np.hstack(scene_lidar_xyz))[0]
    
    sem_map = SemanticMapRef()
    sem_map.xyz_sem_2d = xyz_sem_2d
    sem_map.xyz_sem_3d = xyz_sem_3d
    sem_map.xyz_lidar = xyz_lidar
    sem_map.id = {id: semantic for id, semantic in zip(semantics.id, semantics.semantic)}
    sem_map.voxel_2d = voxel_2d_map.get_bins()
    sem_map.voxel_3d = voxel_3d_map.get_bins()
    
    fn = f"{cfg.hm_data.sem_map_ref}"
    save_obj(filename=fn, obj=sem_map)
    gt_map = load_obj(fn)
    logging.info(f"Saved: {fn}")
    
    
if __name__ == "__main__":
    main()