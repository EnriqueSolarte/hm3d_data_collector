import dvf_map
import hydra
import numpy as np
from imageio import imwrite
import matplotlib.pyplot as plt
import logging
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.io_utils import get_abs_path, create_directory
from geometry_perception_utils.image_utils import get_color_list
import os
from imageio.v3 import imread, imwrite
from matplotlib import pyplot as plt
import json 
import pandas as pd
from dvf_map.experiments.pre_process_hm.utils import read_positions, project_pp_depth, get_cam_pose, mask_semantics


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    
    positions = read_positions(cfg.hm_data.position_fn)
    semantics = pd.read_csv(cfg.hm_data.semantic_fn)
    
    rgb_dir = f"{cfg.hm_data.rgb_dir}"
    semantics_dir = f"{cfg.hm_data.semantic_dir}"
    
    sem_colors = get_color_list(number_of_colors=semantics.shape[0])
    sem_colors = sem_colors[:, np.random.permutation(semantics.shape[0])]
    sem_colors[:, 0] = np.array([0, 0, 0])
     
    for idx in range(len(positions)):
        rgb = imread(f"{rgb_dir}/{idx}.jpg")
        semantics_img = np.load(f"{semantics_dir}/{idx}.npy")        

        semantics_mask = [mask_semantics(semantics_img, id, color) for id, color in zip(semantics.id, sem_colors.T)]
        sem = (np.sum(semantics_mask, axis=0)*255).astype(np.uint8)
        
        comb = np.concatenate([rgb, sem], axis=0)
        fn = f"{cfg.log_dir}/vis.jpg"
        imwrite(f"{fn}", comb.astype(np.uint8))
        print(f"{fn}")
        input("Press Enter to continue...")
    
if __name__ == "__main__":
    main()