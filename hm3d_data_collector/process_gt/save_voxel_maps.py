import hm3d_data_collector
import hydra
import numpy as np
import logging
from hm3d_data_collector.utils.io_utils import get_abs_path, create_directory
import os
import pandas as pd
from hm3d_data_collector.utils.geometry_utils import extend_array_to_homogeneous, eulerAnglesToRotationMatrix
from tqdm import trange
from hm3d_data_collector.utils.data_collection_utils import read_positions, project_pp_depth, get_cam_pose
from hm3d_data_collector.dense_voxel_grid.voxel_grid_2d import VoxelGrid2D
from hm3d_data_collector.dense_voxel_grid.voxel_grid_3d import VoxelGrid3D


def create_voxel_maps(cfg, scene):
    cfg.hm_data.scene, cfg.hm_data.version = scene.split('_')
    logging.info(f"Creating voxel maps for scene at: {cfg.hm_data.saved_dir}")

    positions = read_positions(cfg.hm_data.gt_position_fn)
    semantics = pd.read_csv(cfg.hm_data.semantic_fn)

    voxel2d = VoxelGrid2D(cfg.voxel_grid_2d)
    voxel3d = VoxelGrid3D(cfg.voxel_grid_3d)

    rgb_dir = f"{cfg.hm_data.rgb_dir}"
    depth_dir = f"{cfg.hm_data.depth_dir}"
    semantics_dir = f"{cfg.hm_data.semantic_dir}"

    for idx in trange(len(positions)):
        depth_map = np.load(f"{depth_dir}/{idx}.npy")

        xyz, m = project_pp_depth(depth_map, fov=cfg.hm_data.cfg.habitat.hfov)
        cam_pose = get_cam_pose(positions[idx])

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
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    list_scenes_dir = os.listdir(cfg.data_dir)
    # Loop for all scene defined in the data_dir
    for scene in list_scenes_dir:
        create_voxel_maps(cfg, scene)


if __name__ == "__main__":
    main()
