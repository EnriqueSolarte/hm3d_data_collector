import hydra
from geometry_perception_utils.io_utils import get_abs_path
from hm3d_data_collector.process_gt import process_scene


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    process_scene(cfg)


if __name__ == "__main__":
    main()
