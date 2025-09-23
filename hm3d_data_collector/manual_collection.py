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
import matplotlib.pyplot as plt
import habitat.utils.visualizations.maps as maps
import quaternion as q


def get_top_down_map(sim: Simulator, map_resolution=512):
    top_down_map = maps.get_topdown_map(
        pathfinder=sim.pathfinder,
        map_resolution=map_resolution,
        height=0.0,
        draw_border=True
    )
    return top_down_map


def draw_visualization(sim: Simulator, axs, sensor_view, top_down_map=None):
    if top_down_map is None:
        top_down_map = maps.get_topdown_map(
            pathfinder=sim.pathfinder,
            map_resolution=512,
            height=0.0,
            draw_border=True
        )
    agent_state = sim.get_agent(0).get_state()
    pos_world = np.array(agent_state.position)  # [x, y, z]

    real_x = pos_world[2]
    real_y = pos_world[0]

    forward_vector = q.rotate_vectors(agent_state.rotation, [1, 0, 0])
    forward_xz = np.array([forward_vector[0], 0, forward_vector[2]])
    heading_vector = forward_xz / np.linalg.norm(forward_xz)

    grid_coords = maps.to_grid(
        realworld_x=real_x,
        realworld_y=real_y,
        grid_resolution=top_down_map.shape,
        sim=sim
    )
    [x.clear() for x in axs]
    axs[0].imshow(top_down_map, cmap='gray_r')
    axs[0].scatter(grid_coords[1], grid_coords[0], c='red', s=50)
    axs[0].quiver(
        grid_coords[1], grid_coords[0],
        heading_vector[2], heading_vector[0],
        scale=18, color='blue'
    )
    axs[0].set_aspect('equal')
    
    # view
    axs[1].imshow(sensor_view)
    axs[1].set_axis_off()

    plt.draw()
    plt.pause(0.01)
    plt.show(block=False)


def manual_collection(cfg):
    logging.warning(f"running: {cfg.script}")
    if Path(cfg.hm_data.saved_dir).exists():
        logging.warning(
            f"Directory {cfg.hm_data.saved_dir} already exists. It will be overwritten.")
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
        f"Agent state:\n\tposition: {np.array(agent_state.position)}\n\trotation: {agent_state.rotation}")

    observations = sim.get_sensor_observations()

    # dir save observations
    save_dir = cfg.hm_data.saved_dir

    hm_actions_fn = cfg.hm_data.hm_actions_fn
    open(hm_actions_fn, 'w').close()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    top_down_map = get_top_down_map(sim)
    draw_visualization(sim=sim, top_down_map=top_down_map, axs=axs, sensor_view=observations[color_sensor][:, :, :3])
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
        draw_visualization(sim=sim, top_down_map=top_down_map, axs=axs, sensor_view=observations[color_sensor][:, :, :3])


@hydra.main(version_base=None,
            config_path=HM3D_DATA_COLLECTOR_CFG_DIR,
            config_name="cfg.yaml")
def main(cfg):
    manual_collection(cfg)


if __name__ == "__main__":
    main()
