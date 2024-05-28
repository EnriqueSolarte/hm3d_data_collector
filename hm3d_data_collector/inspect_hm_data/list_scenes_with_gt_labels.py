import dvf_map
import hydra
import numpy as np
import logging
from geometry_perception_utils.io_utils import get_abs_path, create_directory, save_json_dict
from glob import glob 
import json
import logging
import os
        
@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")

    list_sub_dir = cfg.habitat.list_sub_dir
    labeled_scenes = dict()

    for d in list_sub_dir:
        config_fn = glob(f"{cfg.habitat.dataset_dir}/{d}/hm3d_annotated_{d}*")
        hm_config = json.load(open(config_fn[0]))
        
        labeled_scenes[d] = [x.split("/")[0] for x in hm_config['scene_instances']['paths']['.json']]
        logging.info(f"Number of scenes in the split {d}: {labeled_scenes[d].__len__()}")

    fn = f"{os.path.dirname(__file__)}/labeled_scenes.json"
    save_json_dict(filename=fn, dict_data=labeled_scenes)
    logging.info(f"Saved labeled scenes in {fn}")
    
    

if __name__ == "__main__":
    main()