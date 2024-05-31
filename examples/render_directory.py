import hydra
from hm3d_data_collector.utils.io_utils import get_abs_path
from hm3d_data_collector.data_collection import render_directory
import logging
import os


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    render_directory(cfg)


if __name__ == "__main__":
    main()
