import dvf_map
import hydra
import numpy as np
from imageio import imwrite
import matplotlib.pyplot as plt
import logging
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.create_gif import create_gif
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

# methods in this experiment
from dvf_map.experiments.pre_process_hm.utils import read_positions, project_pp_depth, get_cam_pose
from dvf_map.experiments.pre_process_hm.utils import project_pp_depth, get_cam_pose
        


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    
    positions = read_positions(cfg.hm_data.gt_position_fn)
    semantics = pd.read_csv(cfg.hm_data.semantic_fn)
    
    rgb_dir = f"{cfg.hm_data.rgb_dir}"
    depth_dir = f"{cfg.hm_data.depth_dir}"
    semantics_dir = f"{cfg.hm_data.semantic_dir}"
    
    scene_xyz = []
    vis_dir = create_directory(f"{cfg.log_dir}/vis")
    for idx in range(len(positions)):
        rgb = imread(f"{rgb_dir}/{idx}.jpg")
        semantics_img = np.load(f"{semantics_dir}/{idx}.npy")        
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
        
        xyz, m = project_pp_depth(depth_map, fov=cfg.hm_data.cfg.habitat.hfov)
        color_xyz = get_color_array(rgb)[:, m]/255
        cam_pose = get_cam_pose(positions[idx])
        
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz)
        scene_xyz.append(np.vstack((xyz_wc, color_xyz)))
        
        canvas = plot_color_plc(xyz_wc.T, color_xyz.T, return_canvas=True, elevation=90, scale_factor=10, shape=(200, 200))
        img = canvas.render(alpha=True, bgcolor=(0, 0, 0 , 0))
        canvas.close()
        
        # fn = f"{vis_dir}/{idx}.png"
        # imwrite(fn, img)
        # logging.warning(f"saved: {fn}")  
    
    xyz = np.hstack(scene_xyz)
    canvas = plot_color_plc(xyz[:3, :].T, xyz[3:, :].T, return_canvas=True, elevation=90, scale_factor=10, shape=(200, 200))
    img = canvas.render(alpha=True, bgcolor=(0, 0, 0 , 0))
    canvas.close()
    fn = f"{cfg.log_dir}/scene_vis.png"
    imwrite(fn, img)
    logging.warning(f"saved: {fn}")  
    
        
        
if __name__ == "__main__":
    main()