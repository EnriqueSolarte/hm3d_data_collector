import hydra
from geometry_perception_utils.io_utils import get_abs_path
from hm3d_data_collector.data_collection import render_data
import logging
import os


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    list_scenes = os.listdir(cfg.data_dir)
    for render_target_scene in list_scenes:
        logging.info(f"Rendering scene {render_target_scene}")
        cfg.hm_idx_scene = render_target_scene.split('-')[0]
        cfg.hm_scene_name = render_target_scene.split('-')[1].split('_')[0]
        cfg.hm_data_version = render_target_scene.split('-')[1].split('_')[-1]
        render_data(cfg)
        logging.info(f"Scene finished {render_target_scene}")


if __name__ == "__main__":
    main()
