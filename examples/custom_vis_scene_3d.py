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
from hm3d_data_collector.habitat_sim_utils import load_saved_voxel_maps
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.vispy_utils import plot_color_plc
from imageio.v2 import imread
from pathlib import Path
from tqdm import tqdm


def visualize_scene_3d(cfg):
    logging.warning(f"running: {cfg.script}")
    logging.info(
        f"Visualizing voxel maps for scene at: {cfg.hm_data.saved_dir}")

    voxel2d, voxel3d = load_saved_voxel_maps(cfg)

    rgb_dir = f"{cfg.hm_data.rgb_dir}"
    depth_dir = f"{cfg.hm_data.depth_dir}"
    poses_dir = f"{cfg.hm_data.poses_dir}"

    list_idx = [fn.stem for fn in Path(rgb_dir).iterdir()]
    list_idx = sorted(list_idx, key=lambda x: int(x))

    list_xyz = []
    for idx in tqdm(list_idx[0:-1:cfg.get('skip', 10)]):
        rgb = imread(f"{rgb_dir}/{idx}.jpg")
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
        cam_pose = np.load(f"{poses_dir}/{idx}.npy")

        xyz, m = project_pp_depth(depth_map, fov=cfg.habitat.hfov)
        if xyz.shape[1] == 0:
            continue
        xyz_color = get_color_array(rgb)[:, m]/255
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz)
        _ = voxel2d.project_xyz(xyz_wc)
        xyz_wc_vx, vx_idx, xyz_idx, all_xyz_idx = voxel3d.project_xyz(xyz_wc)
        local_xyz = np.vstack([xyz_wc_vx, xyz_color[:, xyz_idx]])
        list_xyz.append(local_xyz)

    xyz_rgb_wc = np.hstack(list_xyz)
    mask = xyz_rgb_wc[1, :] > 0
    plot_color_plc(xyz_rgb_wc[:3, mask].T, xyz_rgb_wc[3:, mask].T)


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    visualize_scene_3d(cfg)


if __name__ == "__main__":
    main()
