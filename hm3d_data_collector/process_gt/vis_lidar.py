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
from dvf_map.ray_tracer.ray_tracer import RaysTracer
from tqdm import tqdm

# methods in this experiment
from dvf_map.experiments.pre_process_hm.utils import read_positions, project_pp_depth, get_cam_pose, project_xyz_to_lidar


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    
    positions = read_positions(cfg.hm_data.gt_position_fn)
    
    rgb_dir = f"{cfg.hm_data.rgb_dir}"
    depth_dir = f"{cfg.hm_data.depth_dir}"
        
    # Ray casting projector
    ray_tracer = RaysTracer(cfg.hm_data.lidar.ray_tracer)

    scene_lidar_xyz = []
    for idx in tqdm(range(len(positions))):
        rgb = imread(f"{rgb_dir}/{idx}.jpg")
        depth_map = np.load(f"{depth_dir}/{idx}.npy")

        xyz, m = project_pp_depth(depth_map, fov=cfg.hm_data.cfg.habitat.hfov)

        xyz_lidar, ret = project_xyz_to_lidar(xyz, ray_tracer, cfg.hm_data.lidar)
        if not ret:
            continue
        cam_pose =  get_cam_pose(positions[idx])
        
        xyz_lidar_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_lidar)
        scene_lidar_xyz.append(xyz_lidar_wc)
    
    canvas = plot_list_pcl(scene_lidar_xyz, return_canvas=True)
    lidar_map = canvas.render()
    fn = f"{cfg.log_dir}/lidar_map.png"
    imwrite(f"{fn}", lidar_map)
    logging.info(f"lidar map saved at {fn}")
        
if __name__ == "__main__":
    main()