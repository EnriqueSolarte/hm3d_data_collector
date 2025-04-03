import hydra
from geometry_perception_utils.io_utils import get_abs_path
from hm3d_data_collector.render import render_hm_data
from hm3d_data_collector.save_voxels_maps import save_voxel_maps


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    render_hm_data(cfg)
    save_voxel_maps(cfg)


if __name__ == "__main__":
    main()
