import hm3d_data_collector
import hydra
import logging
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.io_utils import get_abs_path, create_directory
import os
from geometry_perception_utils.io_utils import save_obj, load_obj
from hm3d_data_collector.process_gt.save_semantic_map import get_semantic_map


def load_saved_semantic_map(cfg, scene):
    cfg.hm_data.scene, cfg.hm_data.version = scene.split('_')
    sem_map_fn = f"{cfg.hm_data.saved_dir}/semantic_map.map"
    if os.path.exists(sem_map_fn):
        logging.info(f"Loading semantic map from: {sem_map_fn}")
        sem_map = load_obj(sem_map_fn)
    else:
        sem_map = get_semantic_map(cfg, scene)
    return sem_map


def prune_sematic_instances(cfg, sem_map):

    EXCLUDED_SEMANTIC = cfg.excluded_semantics

    # Exclude direct instances
    sem = {k: v for k, v in sem_map['idx_layers'].items(
    ) if k.split("_")[-1] not in EXCLUDED_SEMANTIC}

    # exclude instances with less than min_vxl_sem_map voxels
    sem = {k: v for k, v in sem.items() if v.size > cfg.min_vxl_sem_map}

    # exclude compounded instances e.g door frame, picture frame,
    def check_labels(x): return [ex for ex in x.split(
        " ") if ex in EXCLUDED_SEMANTIC].__len__() > 0
    sem = {k: v for k, v in sem.items() if not check_labels(k.split("_")[-1])}

    # Merge instances with the same label
    sem_pruned = {k.split("_")[-1]: {'idx_map': [], 'idx_voxel': []}
                  for k in list(sem.keys())}
    for k, idx_voxel in sem.items():
        idx, label = k.split("_")
        sem_pruned[label]['idx_map'].append(idx)
        sem_pruned[label]['idx_voxel'].append(idx_voxel)
    return {'bins': sem_map['bins'], 'sem_map': sem_pruned}


def process_scene(cfg, scene):
    sem_map = load_saved_semantic_map(cfg, scene)
    sem_map_pruned = prune_sematic_instances(cfg, sem_map)
    fn = f"{cfg.hm_data.saved_dir}/semantic_map_pruned.map"
    save_obj(filename=fn, obj=sem_map_pruned)
    logging.info(f"Semantic map saved at: {fn}")


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    list_scenes_dir = os.listdir(cfg.data_dir)
    # Loop for all scene defined in the data_dir
    for scene in list_scenes_dir:
        process_scene(cfg, scene)


if __name__ == "__main__":
    main()
