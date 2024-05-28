import hydra
from vslab_360_datasets.wrapper_habitat_sim.utils import get_habitat_cfg
from habitat_sim import Simulator
import habitat_sim
import numpy as np
from imageio import imwrite
import logging  
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.io_utils import get_abs_path, create_directory, save_json_dict
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous, eulerAnglesToRotationMatrix
from dvf_map.experiments.pre_process_hm.utils import project_pp_depth, get_cam_pose
from pyquaternion import Quaternion
from omegaconf import OmegaConf, open_dict


def save_semantics(cfg, sim, list_idx):
    semantics = sim.semantic_scene.objects
    
    # Getting registered semantic labels
    metadata = [(int(obj.semantic_id), 
                obj.id.split("_")[0],
                obj.category.index(),
                obj.category.name(),
                obj.region.id.split("_")[-1],5
                )
                for obj in semantics if int(obj.semantic_id) in list_idx]
    
    fn = f"{cfg.data_collection.semantic_fn}"
    f = open(fn, 'w')
    f.write("id,semantic,category_id,category_name,region\n")
    f.close()
    
    with open(fn, 'a') as f:
        [
        f.write(f"{m[0]},{m[1]},{m[2]},{m[3]},{m[4]}\n")
        for m in metadata
        ]
     
    
def read_position(fn):
    with open(fn) as f:
        lines = f.readlines()
    list_position = []
    for line in lines:
        list_position.append([float(x) for x in line.strip().split(',')])
    return list_position


def save_gt_positions(fn, list_position, cfg):
    open(fn, 'w').close()
    sensor_wc = np.eye(4)
    sensor_wc[:3, 3] = cfg.sensor_wc.translation
    R = eulerAnglesToRotationMatrix(np.deg2rad(cfg.sensor_wc.rotation))
    sensor_wc[:3, :3] = R

    cam_0 = get_cam_pose(list_position[0]) @ sensor_wc
    for position in list_position:
        cam_pose = np.linalg.inv(cam_0) @ get_cam_pose(position) @ sensor_wc
        
        with open(fn, 'a') as f:
            trans = cam_pose[:3, 3]
            rotation= cam_pose[:3, :3]
            q = Quaternion(matrix=rotation)
            f.write(f"{','.join([str(x) for x in trans])},")
            f.write(f"{q.x},{q.y},{q.z},{q.w}\n")  

def render(cfg):
    logging.warning(f"running: {cfg.script}")
    
    habitat_cfg = get_habitat_cfg(cfg.habitat)
    
    sim = Simulator(habitat_cfg)
    cfg.data_collection.scene = sim.curr_scene_name
    
    agent = sim.initialize_agent(0)
    
    # Read recorded positions
    list_position = read_position(cfg.data_collection.position_fn)
    # Save GT poses
    save_gt_positions(f"{cfg.data_collection.saved_dir}/gt_poses.txt", list_position, cfg.data_collection)
    
    # Create directories for data 
    rgb_dir = create_directory(f"{cfg.data_collection.saved_dir}/rgb", ignore_request=True)
    depth_dir = create_directory(f"{cfg.data_collection.saved_dir}/depth", ignore_request=True)
    semantic_dir = create_directory(f"{cfg.data_collection.saved_dir}/semantic", ignore_request=True)
    
    color_sensor = cfg.data_collection.color_sensor
    depth_sensor = cfg.data_collection.depth_sensor
    semantic_sensor = cfg.data_collection.semantic_sensor
    
    list_semantic_ids = []
    for idx, position in enumerate(list_position):
        agent_state = habitat_sim.AgentState()
        pos = np.array(position[:3])
        ori = np.quaternion(position[-1], position[3], position[4], position[5])
        agent_state.rotation = ori
        agent_state.position = pos
        agent.set_state(agent_state)

        # Get agent state
        agent_state = agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
        observations = sim.get_sensor_observations()

        img = observations[color_sensor][:, :, :3]
        semantic = observations[semantic_sensor]
        depth = observations[depth_sensor]
        
        # Semantic ids
        mask = depth < cfg.data_collection.max_distance
        sem_ids, counts = np.unique(semantic[mask], return_counts=True)
        sem_ids = sem_ids[counts > cfg.data_collection.min_count_px_semantic]
        [list_semantic_ids.append(idx) for idx in sem_ids if idx not in list_semantic_ids]
        
        imwrite(f"{rgb_dir}/{idx}.jpg", img)        
        np.save(f"{depth_dir}/{idx}.npy", depth)
        np.save(f"{semantic_dir}/{idx}.npy", semantic)
    
    save_semantics(cfg, sim, list_semantic_ids)
    fn = f"{cfg.data_collection.cfg_fn}"
    save_cfg(cfg, cfg_file=fn)
    

@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    render(cfg)
    
if __name__ == "__main__":
    main()