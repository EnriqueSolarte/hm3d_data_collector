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
from geometry_perception_utils.image_utils import get_color_array, get_color_list, get_default_uv_map
from geometry_perception_utils.io_utils import save_obj, load_obj
from dvf_map.experiments.pre_process_hm.utils import (
    read_positions, project_pp_depth, get_cam_pose, 
    project_xyz_to_lidar, mask_semantics, get_bearings)


def get_semantic_map(cfg, scene):
    cfg.hm_data.scene, cfg.hm_data.version = scene.split('_')
    logging.info(f"Processing scene from: {cfg.hm_data.saved_dir}")
    
    fn = f"{cfg.hm_data.saved_dir}/semantic_map.map"
    sem_map = load_obj(fn)
    logging.info(f"Loaded semantic map from: {fn}")
    return sem_map

EXCLUDED_SEMANTIC=['ceiling', "wall", "frame", "rod", "unknown", "Unknown"]

@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    list_scenes_dir = os.listdir(cfg.data_dir)
    # Loop for all scene defined in the data_dir
    for scene in list_scenes_dir:
        
        sem_map = get_semantic_map(cfg, scene)
        bins_3d_map = sem_map['bins']
        voxel_3d_map = VoxelGrid3D.from_bins(u_bins=bins_3d_map[0], v_bins=bins_3d_map[1], c_bins=bins_3d_map[2])
                
        logging.info(f"Saving semantic map for scene: {scene}")
        logging.info(f"Number of layers: {list(sem_map['idx_layers'].keys()).__len__()}")
        logging.info(f"Min number of voxels: {min([v.size for v in sem_map['idx_layers'].values()])}")
        logging.info(f"Max number of voxels: {max([v.size for v in sem_map['idx_layers'].values()])}")
        logging.info(f"Total number of voxels: {sum([v.size for v in sem_map['idx_layers'].values()])}")
        
        data = {k.split("_")[-1]: v.size
        for k, v in sem_map['idx_layers'].items() if k.split("_")[-1] not in EXCLUDED_SEMANTIC}
        
        data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
        
        plt.figure(f"{scene}", figsize=(10, 5), dpi=100, tight_layout=True)
        plt.title(f"{scene}")
        plt.ylabel('Number of voxels')
        plt.xticks(rotation='vertical')
        plt.bar(data.keys(), data.values())
        fn = f"{cfg.hm_data.saved_dir}/his_semantic_map.png"
        plt.savefig(fn)
        logging.info(f"Saved histogram of semantic map to: {fn}")
        
        # for i, vxl_idx in sem_map['idx_layers'].items():
        #     label = str(semantics[semantics.id == i].semantic.values[0])
        #     logging.info(f"Layer: {i} - Sem: {label} - Size: {vxl_idx.size} voxels")
        #     xyz = voxel_3d_map.get_centroids_by_idx(vxl_idx)
        #     plot_list_pcl([xyz])
            

if __name__ == "__main__":
    main()