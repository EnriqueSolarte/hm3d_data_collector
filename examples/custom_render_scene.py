import hydra
from geometry_perception_utils.io_utils import get_abs_path
from hm3d_data_collector.save_voxels_maps import save_voxel_maps
from hm3d_data_collector.habitat_sim_utils import get_habitat_cfg
from habitat_sim import Simulator
import habitat_sim
import numpy as np
from imageio import imwrite
import logging
from geometry_perception_utils.io_utils import create_directory
from hm3d_data_collector.habitat_sim_utils import project_pp_depth, get_cam_pose, get_sensor_wc
from tqdm import tqdm
from geometry_perception_utils.config_utils import save_cfg
import shutil


def render_hm_data(cfg):
    logging.warning(f"running: {cfg.script}")
    save_cfg(cfg, resolve=True)
    habitat_cfg = get_habitat_cfg(cfg.habitat)

    sim = Simulator(habitat_cfg)

    agent = sim.initialize_agent(0)

    # Set initial agent state
    agent_state = habitat_sim.AgentState()
    pos = np.array(cfg.hm_data.init_pos)
    ori = np.quaternion(1, 0, 0, 0)
    agent_state.rotation = ori
    agent_state.position = pos
    agent.set_state(agent_state)

    # Get list of recorded actions
    list_actions = [a for a in open(
        cfg.hm_data.hm_actions_fn).read().split('\n') if a != '']

    # Create directories for data
    rgb_dir = create_directory(
        f"{cfg.hm_data.rgb_dir}", ignore_request=True)
    poses_dir = create_directory(
        f"{cfg.hm_data.poses_dir}", ignore_request=True)
    depth_dir = create_directory(
        f"{cfg.hm_data.depth_dir}", ignore_request=True)
    semantic_dir = create_directory(
        f"{cfg.hm_data.semantic_dir}", ignore_request=True)

    color_sensor = cfg.hm_data.color_sensor
    depth_sensor = cfg.hm_data.depth_sensor
    semantic_sensor = cfg.hm_data.semantic_sensor

    list_semantic_ids = []
    # Get transform from sensor to world
    sensor_wc = get_sensor_wc()
    for idx, action in tqdm(enumerate(list_actions)):
        # agent_state = habitat_sim.AgentState()
        agent.act(action)

        # Get agent state
        agent_state = agent.get_state()
        t = agent_state.sensor_states[color_sensor].position
        q = agent_state.sensor_states[color_sensor].rotation
        if idx == 0:
            cam_pose = np.eye(4)
            initial_pose = get_cam_pose(t, q) @ sensor_wc
        else:
            cam_pose = np.linalg.inv(
                initial_pose) @ get_cam_pose(t, q) @ sensor_wc
        # Save camera pose
        np.save(f"{poses_dir}/{idx}.npy", cam_pose)

        observations = sim.get_sensor_observations()

        img = observations[color_sensor][:, :, :3]
        semantic = observations[semantic_sensor]
        depth = observations[depth_sensor]

        # Semantic ids
        mask = depth < cfg.hm_data.max_distance
        sem_ids, counts = np.unique(semantic[mask], return_counts=True)
        sem_ids = sem_ids[counts > cfg.hm_data.min_count_px_semantic]
        [list_semantic_ids.append(idx)
         for idx in sem_ids if idx not in list_semantic_ids]

        imwrite(f"{rgb_dir}/{idx}.jpg", img)
        np.save(f"{depth_dir}/{idx}.npy", depth)
        np.save(f"{semantic_dir}/{idx}.npy", semantic)

    fn = f"{cfg.hm_data.cfg_fn}"
    shutil.copy(f"{cfg.log_dir}/cfg.yaml", fn)
    logging.info(f"Rendered scene at: {cfg.hm_data.saved_dir}")


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    save_voxel_maps(cfg)


if __name__ == "__main__":
    main()
