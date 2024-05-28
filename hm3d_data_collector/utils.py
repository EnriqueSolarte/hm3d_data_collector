from pyquaternion import Quaternion
import numpy as np
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
# from dvf_map.ray_tracer.ray_tracer import RaysTracer


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


def project_pp_depth(depth_map, mask=None, fov=140):
    """
    Projects depth maps into 3D considering only the pixels in the mask
    """
    
    # bearing vectors
    h, w = depth_map.shape[:2]
    bearings = get_bearings(h, w, fov)
    
    if mask is not None:
        m = mask.flatten() * depth_map.flatten() > 0.2
    else:
        m = depth_map.flatten() > 0.2
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
    return  cam_pose


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

