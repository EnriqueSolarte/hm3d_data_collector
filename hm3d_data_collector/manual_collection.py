from hm3d_data_collector.habitat_sim_utils import get_action
import hydra
from hm3d_data_collector.habitat_sim_utils import get_habitat_cfg
from habitat_sim import Simulator
import habitat_sim
import numpy as np
from imageio import imwrite
import logging
from geometry_perception_utils.io_utils import get_abs_path, create_directory
from geometry_perception_utils.config_utils import save_cfg
from hm3d_data_collector import HM3D_DATA_COLLECTOR_CFG_DIR
from pathlib import Path


def manual_collection(cfg):
    logging.warning(f"running: {cfg.script}")
    if Path(cfg.hm_data.saved_dir).exists():
        logging.warning(f"Directory {cfg.hm_data.saved_dir} already exists. It will be overwritten.")
        input("Press Enter to continue or Ctrl+C to exit...")
    save_cfg(cfg, resolve=True)
    habitat_cfg = get_habitat_cfg(cfg.habitat)

    sim = Simulator(habitat_cfg)
    agent = sim.initialize_agent(0)

    # Defining sensor
    color_sensor = cfg.hm_data.color_sensor
    depth_sensor = cfg.hm_data.depth_sensor

    # Set agent state
    agent_state = habitat_sim.AgentState()
    pos = np.array(cfg.hm_data.init_pos)
    ori = np.quaternion(cfg.hm_data.init_rot[0],
                        cfg.hm_data.init_rot[1],
                        cfg.hm_data.init_rot[2],
                        cfg.hm_data.init_rot[3])
    agent_state.rotation = ori
    agent_state.position = pos
    agent.set_state(agent_state)

    # Get agent state
    agent_state = agent.get_state()
    logging.info(
        f"Agent state:\n\tposition: {agent_state.position}\n\trotation: {agent_state.rotation}")

    observations = sim.get_sensor_observations()

    # dir save observations
    save_dir = cfg.hm_data.saved_dir

    hm_actions_fn = cfg.hm_data.hm_actions_fn
    open(hm_actions_fn, 'w').close()
    while True:
        # Get sensor observations
        img = observations[color_sensor][:, :, :3]
        depth = observations[depth_sensor]

        # Save observations
        depth = 255 * depth / depth.max()
        comb = img
        # comb = np.vstack(
        #     [img,  np.repeat(depth[:, :, None], 3, axis=-1)]).astype(np.uint8)
        # fn = f"{save_dir}/{idx}.jpg"
        fn = f"{save_dir}/visualization.jpg"
        imwrite(fn, comb)
        print(f"For visualization see: {fn}")

        action = get_action()
        agent.act(action)

        agent_state = agent.get_state()

        position = agent_state.sensor_states[color_sensor].position
        rotation = agent_state.sensor_states[color_sensor].rotation
        logging.info(f"Action: {action}")
        logging.info(
            f"Agent state:\n\tposition: {position}\n\trotation: {rotation}")

        with open(hm_actions_fn, 'a') as f:
            f.write(f"{action}\n")
        observations = sim.get_sensor_observations()


@hydra.main(version_base=None,
            config_path=HM3D_DATA_COLLECTOR_CFG_DIR,
            config_name="cfg.yaml")
def main(cfg):
    manual_collection(cfg)


if __name__ == "__main__":
    main()
