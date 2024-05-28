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

    
@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    
    fn = f"{cfg.hm_data.sem_map_ref}"
    sem_map = load_obj(fn)
    logging.info(f"loaded: {fn}")
    
    # plot 3d map 
    plot_list_pcl([xyz for xyz in sem_map.xyz_sem_3d.values()])
    
    # plot 2d map
    plot_list_pcl([xyz for xyz in sem_map.xyz_sem_3d.values()])
    
    # plot lidar 
    plot_list_pcl([sem_map.xyz_lidar])
    
    
if __name__ == "__main__":
    main()