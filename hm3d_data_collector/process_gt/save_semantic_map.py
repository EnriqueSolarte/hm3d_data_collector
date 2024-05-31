import hm3d_data_collector
import hydra
import numpy as np
import logging
from hm3d_data_collector.utils.io_utils import get_abs_path
import os
import pandas as pd
from hm3d_data_collector.utils.geometry_utils import extend_array_to_homogeneous
from hm3d_data_collector.utils.image_utils import get_color_array
from tqdm import trange
from hm3d_data_collector.utils.data_collection_utils import read_positions, project_pp_depth, get_cam_pose
from hm3d_data_collector.utils.data_collection_utils import project_pp_depth, get_cam_pose
from hm3d_data_collector.dense_voxel_grid.voxel_grid_2d import VoxelGrid2D
from hm3d_data_collector.dense_voxel_grid.voxel_grid_3d import VoxelGrid3D
from hm3d_data_collector.utils.image_utils import get_color_array, get_color_list, get_default_uv_map
from hm3d_data_collector.utils.io_utils import save_obj, load_obj
from hm3d_data_collector.process_gt.save_voxel_maps import create_voxel_maps


def register_semantic_id(local_data, layered_map):
    for i, vxl_idx in local_data.items():
        if i in layered_map.keys():
            layered_map[i] = layered_map[i].union(set(vxl_idx))
        else:
            layered_map[i] = set(vxl_idx)


def load_saved_voxel_maps(cfg, scene):
    # Voxel map for the data
    voxel_2d_map_fn = f"{cfg.hm_data.saved_dir}/bins_voxel_map_2d.npy"
    voxel_3d_map_fn = f"{cfg.hm_data.saved_dir}/bins_voxel_map_3d.npy"
    if not os.path.exists(voxel_2d_map_fn) or not os.path.exists(voxel_3d_map_fn):
        create_voxel_maps(cfg, scene)

    bins_2d_map = np.load(f"{voxel_2d_map_fn}", allow_pickle=True)
    bins_3d_map = np.load(f"{voxel_3d_map_fn}",  allow_pickle=True)

    voxel_2d_map = VoxelGrid2D.from_bins(
        u_bins=bins_2d_map[0], v_bins=bins_2d_map[1])
    voxel_3d_map = VoxelGrid3D.from_bins(
        u_bins=bins_3d_map[0], v_bins=bins_3d_map[1], c_bins=bins_3d_map[2])

    logging.info(f"Loaded 2D voxel map: {voxel_2d_map.shape}")
    logging.info(f"Loaded 3D voxel map: {voxel_3d_map.shape}")
    return voxel_2d_map, voxel_3d_map


def get_semantic_map(cfg, scene):
    cfg.hm_data.scene, cfg.hm_data.version = scene.split('_')
    logging.info(
        f"Creating GT semantic map for scene at: {cfg.hm_data.saved_dir}")

    EXCLUDED_SEMANTIC = cfg.excluded_semantics

    # loading pre-computed bins
    voxel_2d_map, voxel_3d_map = load_saved_voxel_maps(cfg, scene)

    positions = read_positions(cfg.hm_data.gt_position_fn)
    semantics = pd.read_csv(cfg.hm_data.semantic_fn)
    sem_labels = {i: str(
        semantics[semantics.id == i].semantic.values[0]) for i in list(semantics.id)}
    sem_labels = {k: v for k, v in sem_labels.items()
                  if v not in EXCLUDED_SEMANTIC}

    depth_dir = f"{cfg.hm_data.depth_dir}"
    semantics_dir = f"{cfg.hm_data.semantic_dir}"

    layered_3d_map = {}

    # To avoid re-accessing to OmegaConf
    fov = cfg.hm_data.cfg.habitat.hfov
    ratio = cfg.hm_data.semantic_ratio
    for idx in trange(len(positions)):
        # for idx in trange(10):
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
        semantics_map = np.load(f"{semantics_dir}/{idx}.npy")
        semantics_ids = get_color_array(semantics_map)[0]

        # xyz in CC (from depth map to Euclidean space)
        dist_mask = depth_map < 3
        xyz_cc, xyz_mask = project_pp_depth(depth_map, fov=fov, mask=dist_mask)

        cam_pose = get_cam_pose(positions[idx])
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_cc)

        xyz_wc_vx, vxl_idx, xyz_idx, all_xyz_idx = voxel_3d_map.project_xyz(
            xyz_wc)

        # Only suitable semantic ids that are in the voxel map
        masked_semantics_ids = semantics_ids[xyz_mask][xyz_idx]

        # all id semantics must be in the voxel map
        assert masked_semantics_ids.shape[0] == xyz_idx.shape[0], "Semantic and xyz should have the same length"

        unique_id, counts_id = np.unique(
            masked_semantics_ids, return_counts=True)

        # suitable unique id
        suitable_unique_id = unique_id[counts_id > xyz_idx.size * ratio]
        # plot_list_pcl([xyz_wc_vx[:, masked_semantics_ids == i] for i in suitable_unique_id])

        local_data = {f"{idx}_{sem_labels[idx]}": set(vxl_idx[masked_semantics_ids == idx])
                      for idx in suitable_unique_id if idx in sem_labels.keys()}

        register_semantic_id(local_data, layered_3d_map)

    # semantic map
    sem_map = {'bins': voxel_3d_map.get_bins()}
    sem_map["idx_layers"] = {k: np.array(list(v))
                             for k, v in layered_3d_map.items()}
    return sem_map


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    list_scenes_dir = os.listdir(cfg.data_dir)
    # Loop for all scene defined in the data_dir
    for scene in list_scenes_dir:
        sem_map = get_semantic_map(cfg, scene)

        logging.info(f"Saving semantic map for scene: {scene}")
        logging.info(
            f"Number of layers: {list(sem_map['idx_layers'].keys()).__len__()}")
        logging.info(
            f"Min number of voxels: {min([v.size for v in sem_map['idx_layers'].values()])}")
        logging.info(
            f"Max number of voxels: {max([v.size for v in sem_map['idx_layers'].values()])}")
        logging.info(
            f"Total number of voxels: {sum([v.size for v in sem_map['idx_layers'].values()])}")

        fn = f"{cfg.hm_data.saved_dir}/semantic_map.map"
        save_obj(filename=fn, obj=sem_map)
        logging.info(f"Semantic map saved at: {fn}")


if __name__ == "__main__":
    main()
