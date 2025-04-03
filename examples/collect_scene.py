import hydra
from geometry_perception_utils.io_utils import get_abs_path
from hm3d_data_collector.manual_collection import manual_collection


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    manual_collection(cfg)


if __name__ == "__main__":
    main()
