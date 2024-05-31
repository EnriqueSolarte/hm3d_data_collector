import hm3d_data_collector
import hydra
import logging
from hm3d_data_collector.utils.io_utils import get_abs_path, create_directory, save_json_dict
from hm3d_data_collector.data_collection.render import render


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    render_target_scene = cfg.target_scene
    logging.info(f"Rendering scene {render_target_scene}")
    cfg.hm_idx_scene = render_target_scene.split('-')[0]
    cfg.hm_scene_name = render_target_scene.split('-')[1].split('_')[0]
    cfg.hm_data_version = render_target_scene.split('-')[1].split('_')[-1]
    render(cfg)
    logging.info(f"Scene finished {render_target_scene}")


if __name__ == "__main__":
    main()
