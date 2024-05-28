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
from geometry_perception_utils.image_utils import get_color_array, get_color_list
from geometry_perception_utils.vispy_utils import plot_list_pcl, plot_color_plc
from geometry_perception_utils.io_utils import save_obj, load_obj
from tqdm import trange
# methods in this experiment
from dvf_map.experiments.pre_process_hm.utils import read_positions, project_pp_depth, get_cam_pose
from vis_semantics import mask_semantics
from dvf_map.experiments.pre_process_hm.utils import project_pp_depth, get_cam_pose
from dvf_map.dense_voxel_grid.voxel_grid_2d import VoxelGrid2D, BEVMap

        
        
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
    
    sem_colors = get_color_list(number_of_colors=semantics.shape[0])
    sem_colors = sem_colors[:, np.random.permutation(semantics.shape[0])]
    sem_colors[:, 0] = np.array([0, 0, 0])
    
    voxel_map = VoxelGrid2D(cfg.voxel_grid_2d)
    dict_bev_maps = {id: [] for id in semantics['id']}
    for idx in trange(len(positions)):
        # Process xyz data
        depth_map = np.load(f"{depth_dir}/{idx}.npy")

        xyz_cc, m = project_pp_depth(depth_map, fov=cfg.hm_data.cfg.habitat.hfov)
        cam_pose = get_cam_pose(positions[idx])
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_cc)

        semantics_img = np.load(f"{semantics_dir}/{idx}.npy")
        # Each item corresponds to a each category
        list_semantic_maps = {id: mask_semantics(semantics_img, id, color) for id, color in zip(semantics.id, sem_colors.T)}
        
        for id, sem in list_semantic_maps.items() :
            # avoid empty semantic masks
            if np.sum(sem) == 0:
                continue
            semantic_xyz = get_color_array(sem)[:, m]
            semantic_mask = np.sum(semantic_xyz, axis=0) > 0
            
            # plot_color_plc(xyz_wc[:, semantic_mask].T, semantic_xyz[:, semantic_mask].T)
            
            xyz_wc_vx, xyz_idx, vxl_idx = voxel_map.project_xyz(xyz_wc[:, semantic_mask])
            dict_bev_maps[id].append(xyz_wc_vx)

    # Aggregate semantic from multi-view
    dict_bev_maps = {id: np.hstack(dict_bev_maps[id]) for id in dict_bev_maps.keys() if id != 0}
    
    # Semantic voxel map
    dict_bev_maps = {id: voxel_map.project_xyz(dict_bev_maps[id])[0] for id in dict_bev_maps.keys() if id != 0}

    fn = f"{cfg.log_dir}/{cfg.hm_data.scene}_{cfg.hm_data.version}.bev_gt_map"
    save_obj(filename=fn, obj=dict_bev_maps)
    
    gt_map = load_obj(fn)
    
    logging.info(f"Saved: {fn}")
    canvas = plot_list_pcl([x for x in dict_bev_maps.values()], return_canvas=True)
    img = canvas.render()
    fn = f"{cfg.log_dir}/{cfg.hm_data.scene}_{cfg.hm_data.version}.bev_gt_map.png"
    imwrite(f"{fn}", img)
    logging.info(f"Saved: {fn}")
    
        
if __name__ == "__main__":
    main()