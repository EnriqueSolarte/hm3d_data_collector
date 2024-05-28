import dvf_map
import hydra
import numpy as np
from imageio import imwrite
import matplotlib.pyplot as plt
import logging
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.io_utils import get_abs_path, create_directory, save_csv_file
from geometry_perception_utils.image_utils import get_color_list
import os
from imageio.v3 import imread, imwrite
from matplotlib import pyplot as plt
import json 
import pandas as pd
from dvf_map.experiments.pre_process_hm.utils import read_positions, project_pp_depth, get_cam_pose, mask_semantics
import clip
import csv
import torch
from tqdm import tqdm


def collect_all_semantics(cfg):
    list_scenes_dir = os.listdir(cfg.data_dir)
    semantics = {}
    for scenes in list_scenes_dir:
        cfg.scene_name, cfg.scene_version = scenes.split('_')
        logging.info(f"Processing scene from: {cfg.hm_data.saved_dir}")
        
        df_semantics = pd.read_csv(cfg.hm_data.semantic_fn)
        semantics[scenes] = list(set(df_semantics['semantic']))

    return semantics


def print_semantic_stats(sem_dict: dict):
    for scene, list_sem in sem_dict.items():
        logging.info(f"Scene: {scene}")
        logging.info(f"Semantics found: {list_sem.__len__()}")


def save_stable_categories(cfg):
    """
    We pre-define a stable category as the categories which have at least 
    cfg.ratio of pixels in each collected frame. This list categories will be 
    saved in a file prune_semantics.txt for each scene.
    """
    
    list_scenes_dir = os.listdir(cfg.data_dir)
    for scene in list_scenes_dir:
        cfg.scene_name, cfg.scene_version = scene.split('_')
        logging.info(f"Processing scene from: {cfg.hm_data.saved_dir}")
    
        semantics_dir = f"{cfg.hm_data.semantic_dir}"
        positions = read_positions(cfg.hm_data.position_fn)
        df_semantics = pd.read_csv(cfg.hm_data.semantic_fn)
        all_ids = list(set(df_semantics['id']))
        logging.info(f"All categories: {all_ids.__len__()}")
        
        list_stable_ids = []
        for idx in tqdm(range(len(positions)), desc="Processing stable semantics"):
            semantics_img = np.load(f"{semantics_dir}/{idx}.npy")        

            id_sem, count = np.unique(semantics_img, return_counts=True)
            mask = count > cfg.hm_data.stable_semantic_ratio*semantics_img.size
            
            stable_id = id_sem[mask]
            [list_stable_ids.append(i) for i in stable_id]
        
        list_stable_ids = list(set(list_stable_ids))
        logging.info(f"Stable categories: {list_stable_ids.__len__()}")
        
        fn = f"{cfg.hm_data.saved_dir}/prune_semantics.txt"
        
        with open(fn, "+w") as f:
            writer = csv.writer(f)
            [writer.writerow([l]) for l in list_stable_ids]
        
        logging.info(f"Saved stable categories to: {fn}")
        
    

@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    
    semantics = collect_all_semantics(cfg)
    print_semantic_stats(semantics)
    
    save_stable_categories(cfg)
    
    
    
    
    
if __name__ == "__main__":
    main()