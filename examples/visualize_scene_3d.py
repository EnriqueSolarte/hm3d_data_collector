import hydra
from geometry_perception_utils.io_utils import get_abs_path
from hm3d_data_collector.visualize_scene_3d import visualize_scene_3d


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    visualize_scene_3d(cfg)


if __name__ == "__main__":
    main()
