from pyquaternion import Quaternion
from getch import getch
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous, eulerAnglesToRotationMatrix
from habitat_sim.utils.settings import default_sim_settings
from habitat_sim.utils.settings import make_cfg
from habitat_sim import Simulator
import numpy as np
import os
import habitat_sim
from geometry_perception_utils.dense_voxel_grid.voxel_grid_2d import VoxelGrid2D
from geometry_perception_utils.dense_voxel_grid.voxel_grid_3d import VoxelGrid3D


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


def get_sensor_wc(t=np.array((0, 0, 0)), rpy=(np.deg2rad(180), 0, 0)):
    sensor_wc = np.eye(4)
    R = eulerAnglesToRotationMatrix(rpy)
    sensor_wc[:3, :3] = R
    sensor_wc[:3, 3] = t
    return sensor_wc


def load_saved_voxel_maps(cfg):
    # Voxel map for the data
    voxel_2d_map_fn = f"{cfg.hm_data.bins_voxel_map_2d_fn}"
    voxel_3d_map_fn = f"{cfg.hm_data.bins_voxel_map_3d_fn}"
    if not os.path.exists(voxel_2d_map_fn):
        raise FileNotFoundError(
            f"Voxel map file not found: {voxel_2d_map_fn}")
    if not os.path.exists(voxel_3d_map_fn):
        raise FileNotFoundError(
            f"Voxel map file not found: {voxel_3d_map_fn}")

    bins_2d_map = np.load(f"{voxel_2d_map_fn}", allow_pickle=True)
    bins_3d_map = np.load(f"{voxel_3d_map_fn}",  allow_pickle=True)

    voxel_2d_map = VoxelGrid2D.from_bins(
        u_bins=bins_2d_map[0], v_bins=bins_2d_map[1])
    voxel_3d_map = VoxelGrid3D.from_bins(
        u_bins=bins_3d_map[0], v_bins=bins_3d_map[1], c_bins=bins_3d_map[2])

    return voxel_2d_map, voxel_3d_map


def get_cam_pose(translation, quaternion):
    """
    Returns the camera pose given the agent states.
    """
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = translation
    q = Quaternion(x=quaternion.x, y=quaternion.y,
                   z=quaternion.z, w=quaternion.w)
    cam_pose[:3, :3] = q.rotation_matrix
    return cam_pose


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


set_actions = dict(
    q='move_up',
    c='move_down',
    i='look_up',
    k='look_down',
    w='move_forward',
    s='move_backward',
    a='move_left',
    d='move_right',
    j='turn_left',
    l='turn_right',
)


def get_action():
    [print(f"\t|\tPress key '{k}' to move the agent '{m}'")
     for k, m in set_actions.items()]
    while True:
        key = getch()
        pressed = [m for k, m in set_actions.items() if k == key]
        if pressed.__len__() > 0:
            action = pressed[0]
            break
    return action


def get_bearings(h, w, fov):
    fx, fy = np.deg2rad(fov), np.deg2rad(fov*h/w)
    x = np.linspace(-np.tan(fx/2), np.tan(fx/2), w)
    y = np.linspace(-np.tan(fy/2), np.tan(fy/2), h)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack((xx.flatten(), yy.flatten()))
    bearings = extend_array_to_homogeneous(xy)
    return bearings


def project_pp_depth(depth_map, mask=None, fov=140, min_depth=0.2):
    """
    Projects depth maps into 3D considering only the pixels in the mask
    """

    # bearing vectors
    h, w = depth_map.shape[:2]
    bearings = get_bearings(h, w, fov)

    if mask is not None:
        m = mask.flatten()
    else:
        m = depth_map.flatten() > min_depth
    # m = depth_map.flatten() > 0.2
    # scale = depth_map.flatten()[m] /np.abs(bearings[0, m])
    xyz = depth_map.flatten()[m] * bearings[:, m]
    return xyz, m


def mask_semantics(img, id, color):
    output = np.zeros(img.shape + (3,))
    output[img == id] = color
    return output


def project_xyz_to_lidar(xyz, ray_tracer, cfg):
    # Masking slide for mimicking the lidar data
    mask = np.abs(xyz[1, :] - cfg.offset_height) < cfg.slide_bandwidth
    if np.sum(mask) == 0:
        return None, False

    xyz_lidar_slide = xyz[:, mask]
    xyz_lidar_rays = ray_tracer.project_on_rays(xyz_lidar_slide)

    if xyz_lidar_rays.size == 0:
        return None, False

    return xyz_lidar_rays, True
