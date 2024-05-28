import hydra
from vslab_360_datasets.wrapper_habitat_sim.utils import get_habitat_cfg
from habitat_sim import Simulator
import habitat_sim
import numpy as np
from imageio import imwrite
import matplotlib.pyplot as plt
import logging
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.io_utils import get_abs_path, create_directory
from geometry_perception_utils.geometry_utils import eulerAnglesToRotationMatrix
from getch import getch
from pyquaternion import Quaternion

        
def get_action():
    print("w: move forward, a: turn left, d: turn right")
    while True:
        key = getch()
        if key == "q":
            action = "move_up"
            break
        if key == "c":
            action = "move_down"
            break
        if key == "i":
            action = "look_up"
            break
        if key == "k":
            action = "look_down"
            break
        if key == "w":
            action = "move_forward"
            break
        if key == "s":
            action = "move_backward"
            break
        if key == "a":
            action = "move_left"
            break
        if key == "d":
            action = "move_right"
            break
        if key == "j":
            action = "turn_left"
            break
        if key == "l":
            action = "turn_right"    
            break
    return action

    
@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning(f"running: {cfg.script}")
    # save_cfg(cfg)
    
    habitat_cfg = get_habitat_cfg(cfg.habitat)
    
    sim = Simulator(habitat_cfg)
    agent = sim.initialize_agent(0)
    
    # Defining sensor 
    color_sensor = cfg.data_collection.color_sensor
    depth_sensor = cfg.data_collection.depth_sensor
    
    # Set agent state
    agent_state = habitat_sim.AgentState()
    pos = np.array(cfg.data_collection.init_pos)
    ori = np.quaternion(1, 0, 0, 0)
    agent_state.rotation = ori
    agent_state.position = pos
    agent.set_state(agent_state)

    # Get agent state
    agent_state = agent.get_state()
    print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
    observations = sim.get_sensor_observations()

    # Create directory to save observations
    cfg.data_collection.scene = sim.curr_scene_name
    save_dir = create_directory(cfg.data_collection.saved_dir)
    
    hm_states_fn = cfg.data_collection.position_fn
    open(hm_states_fn, 'w').close()
    while True:
        # Get sensor observations
        img = observations[color_sensor][:, :, :3]
        depth = observations[depth_sensor]
        
        # Save observations
        depth = 255 * depth / depth.max()
        comb = np.vstack([img,  np.repeat(depth[:, :, None], 3, axis=-1)]).astype(np.uint8)
        # fn = f"{save_dir}/{idx}.jpg"
        fn = f"{save_dir}/visualization.jpg"
        imwrite(fn, comb)
        print(f"Saved: {fn}")
        
        action = get_action()
        agent.act(action)
        
        agent_state = agent.get_state()
        position = agent_state.position
        rotation = agent_state.rotation
        print("agent_state: position", position, "rotation", rotation)
        print(f"position: {', '.join([str(v) for v in list(agent_state.position)])}")
        with open(hm_states_fn, 'a') as f:
            f.write(f"{','.join([str(x) for x in position])},")
            ori = rotation
            f.write(f"{ori.x},{ori.y},{ori.z},{ori.w}\n")  
        observations = sim.get_sensor_observations()


if __name__ == "__main__":
    main()