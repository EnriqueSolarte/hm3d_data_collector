from hm3d_data_collector import HM3D_DATA_COLLECTOR_CFG_DIR
import hydra
import numpy as np
import logging
from geometry_perception_utils.io_utils import get_abs_path, create_directory
import os
import pandas as pd
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous, eulerAnglesToRotationMatrix
from tqdm import trange
from hm3d_data_collector.habitat_sim_utils import project_pp_depth, get_cam_pose
from geometry_perception_utils.dense_voxel_grid.voxel_grid_2d import VoxelGrid2D
from geometry_perception_utils.dense_voxel_grid.voxel_grid_3d import VoxelGrid3D
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.vispy_utils import plot_color_plc
from imageio.v2 import imread
from pathlib import Path
from tqdm import tqdm


def save_voxel_maps(cfg):
    logging.warning(f"running: {cfg.script}")
    logging.info(
        f"Saving voxel maps for scene at: {cfg.hm_data.saved_dir}")

    voxel2d = VoxelGrid2D(cfg.voxel_grid_2d)
    voxel3d = VoxelGrid3D(cfg.voxel_grid_3d)

    rgb_dir = f"{cfg.hm_data.rgb_dir}"
    depth_dir = f"{cfg.hm_data.depth_dir}"
    poses_dir = f"{cfg.hm_data.poses_dir}"

    list_idx = [fn.stem for fn in Path(rgb_dir).iterdir()]
    list_idx = sorted(list_idx, key=lambda x: int(x))

    for idx in tqdm(list_idx):
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
        cam_pose = np.load(f"{poses_dir}/{idx}.npy")

        xyz, m = project_pp_depth(depth_map, fov=cfg.habitat.hfov)

        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz)
        _ = voxel2d.project_xyz(xyz_wc)
        _ = voxel3d.project_xyz(xyz_wc)

    fn = f"{cfg.hm_data.saved_dir}/bins_voxel_map_2d.npy"
    np.save(fn, np.asarray(voxel2d.get_bins(), dtype=object))
    logging.info(f"2D voxel map saved @ {fn}")
    fn = f"{cfg.hm_data.saved_dir}/bins_voxel_map_3d.npy"
    np.save(fn, np.asarray(voxel3d.get_bins(), dtype=object))
    logging.info(f"3D voxel map saved @ {fn}")


@hydra.main(version_base=None,
            config_path=HM3D_DATA_COLLECTOR_CFG_DIR,
            config_name="cfg.yaml")
def main(cfg):
    
    save_voxel_maps(cfg)


if __name__ == "__main__":
    main()
