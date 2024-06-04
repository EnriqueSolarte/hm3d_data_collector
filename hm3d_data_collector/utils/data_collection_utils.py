from pyquaternion import Quaternion
import numpy as np
from hm3d_data_collector.utils.geometry_utils import extend_array_to_homogeneous
from getch import getch

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


def read_positions(fn):
    with open(fn) as f:
        lines = f.readlines()
    list_position = []
    for line in lines:
        list_position.append([float(x) for x in line.strip().split(',')])
    return list_position


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


def get_cam_pose(position):
    """
    Returns the camera pose given the position.
    Replace this with the actual implementation of getting the camera pose.
    """
    # Replace this with the actual implementation of getting the camera pose

    cam_pose = np.eye(4)
    cam_pose[:3, 3] = position[:3]
    q = Quaternion(x=position[3], y=position[4], z=position[5], w=position[6])
    cam_pose[:3, :3] = q.rotation_matrix
    return cam_pose


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
