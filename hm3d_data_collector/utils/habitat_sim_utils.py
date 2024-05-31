from habitat_sim.utils.settings import default_sim_settings
from habitat_sim.utils.settings import make_cfg
from habitat_sim import Simulator
import habitat_sim.utils.viz_utils
import numpy as np
import os


def get_habitat_cfg(habitat_params: dict):
    """
    Creates a habitat config from a dictionary of habitat parameters.
    """

    os.environ['GLOG_minloglevel'] = "3"
    os.environ['MAGNUM_LOG'] = "quiet"
    os.environ['HABITAT_SIM_LOG'] = "quiet"

    cfg = default_sim_settings.copy()
    cfg.update(habitat_params)
    cfg = make_cfg(cfg)
    cfg.agents[0].action_space.clear()

    for action in habitat_params.actions.items():
        cfg.agents[0].action_space.setdefault(
            action[0], habitat_sim.agent.ActionSpec(
                action[0], habitat_sim.agent.ActuationSpec(amount=action[1])))

    return cfg


def get_random_initial_and_goal(sim: Simulator, params: dict):
    # Reading hyperparameters
    min_path_distance = params.min_path_distance
    seed = params.seed

    sim.pathfinder.seed(seed)

    path = habitat_sim.ShortestPath()

    path.requested_start = sim.pathfinder.get_random_navigable_point()
    while True:
        path.requested_end = sim.pathfinder.get_random_navigable_point()
        found = sim.pathfinder.find_path(path)
        if found and path.geodesic_distance > min_path_distance:
            break

    return path.requested_start, path.requested_end, found


def get_list_actions(initial_point, goal, sim: Simulator, agent):

    # define follower
    follower = habitat_sim.GreedyGeodesicFollower(
        sim.pathfinder,
        agent,
        forward_key="move_forward",
        left_key="turn_left",
        right_key="turn_right",
    )

    # Define initial state
    state = habitat_sim.AgentState()
    state.position = initial_point
    state.rotation = np.quaternion(1, 0, 0, 0)
    agent.set_state(state)

    try:
        action_list = follower.find_path(goal)
    except habitat_sim.errors.GreedyFollowerError:
        action_list = [None]
    return action_list
