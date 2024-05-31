import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
import dill
import numpy as np
from plyfile import PlyData
from pyquaternion import Quaternion
from multiprocessing.pool import ThreadPool


def load_instances(self, cfg):
    """
    Load key-value pairs from a config file into the class namespace.
    """
    [setattr(self, key, val) for key, val in cfg.items()]


def save_json_dict(filename, dict_data):
    with open(filename, "w") as outfile:
        json.dump(dict_data, outfile, indent="\t")


def read_txt_file(filename):

    with open(filename, "r") as fn:
        data = fn.read().splitlines()

    return data


def read_csv_file(filename):
    with open(filename) as f:
        csvreader = csv.reader(f)

        lines = []
        for row in csvreader:
            lines.append(row[0])
    return lines


def save_csv_file(filename, data, flag="w"):
    with open(filename, flag) as f:
        writer = csv.writer(f)
        for line in data:
            writer.writerow([l for l in line])
    f.close()


def load_obj(filename):
    return dill.load(open(filename, "rb"))


def create_directory(output_dir, delete_prev=True, ignore_request=False):
    if os.path.exists(output_dir) and delete_prev:
        if not ignore_request:
            logging.warning(f"This directory will be deleted: {output_dir}")
            input("This directory will be deleted. PRESS ANY KEY TO CONTINUE...")
        shutil.rmtree(output_dir, ignore_errors=True)
    if not os.path.exists(output_dir):
        logging.info(f"Dir created: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    return Path(output_dir).resolve().__str__()


def save_obj(filename, obj):
    dill.dump(obj, open(filename, "wb"))
    print(f" >> OBJ saved: {filename}")


def get_files_given_a_pattern(
    data_dir, flag_file, exclude="", include_flag_file=False, isDir=False
):
    """
    Searches in the passed @data_dir, recurrently, the sub-directories which content the @flag_file,
    excluding directories listed in @exclude.
    """
    scenes_paths = []
    for root, dirs, files in tqdm(
        os.walk(data_dir), desc=f"Walking through {data_dir}..."
    ):
        dirs[:] = [d for d in dirs if d not in exclude]
        if not isDir:
            if include_flag_file:
                [
                    scenes_paths.append(os.path.join(root, f))
                    for f in files
                    if flag_file in f
                ]
            else:
                [scenes_paths.append(root) for f in files if flag_file in f]
        else:
            [
                scenes_paths.append(os.path.join(root, flag_file))
                for d in dirs
                if flag_file in d
            ]

    return scenes_paths


def mytransform44(l, seq="xyzw"):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    if seq == "wxyz":
        if q[0] < 0:
            q *= -1
        q = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    else:
        if q[3] < 0:
            q *= -1
        q = Quaternion(
            x=q[0],
            y=q[1],
            z=q[2],
            w=q[3],
        )
    trasnform = np.eye(4)
    trasnform[0:3, 0:3] = q.rotation_matrix
    trasnform[0:3, 3] = np.array(t)

    return trasnform


def read_trajectory(filename, matrix=True, traj_gt_keys_sorted=[], seq="xyzw"):
    """
    Read a trajectory from a text file.

    Input:
    filename -- file to be read_datasets
    matrix -- convert poses to 4x4 matrices

    Output:
    dictionary of stamped 3D poses
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [
        [float(v.strip()) for v in line.split(" ") if v.strip() != ""]
        for line in lines
        if len(line) > 0 and line[0] != "#"
    ]
    list_ok = []
    for i, l in enumerate(list):
        if l[4:8] == [0, 0, 0, 0]:
            continue
        isnan = False
        for v in l:
            if np.isnan(v):
                isnan = True
                break
        if isnan:
            sys.stderr.write(
                "Warning: line {} of file {} has NaNs, skipping line\n".format(
                    i, filename
                )
            )
            continue
        list_ok.append(l)
    if matrix:
        traj = dict([(l[0], mytransform44(l[0:], seq=seq)) for l in list_ok])
    else:
        traj = dict([(l[0], l[1:8]) for l in list_ok])

    return traj


def read_json_label(fn):
    with open(fn, "r") as f:
        d = json.load(f)
        room_list = d["room_corners"]
        room_corners = []
        for corners in room_list:
            corners = np.asarray([[float(x[0]), float(x[1])] for x in corners])
            room_corners.append(corners)
        axis_corners = d["axis_corners"]
        if axis_corners.__len__() > 0:
            axis_corners = np.asarray(
                [[float(x[0]), float(x[1])] for x in axis_corners]
            )
    return room_corners, axis_corners


def read_ply(fn):
    plydata = PlyData.read(fn)
    v = np.array([list(x) for x in plydata.elements[0]])
    points = np.ascontiguousarray(v[:, :3])
    points[:, 0:3] = points[:, [0, 2, 1]]
    colors = np.ascontiguousarray(v[:, 6:9], dtype=np.float32) / 255
    return np.concatenate((points, colors), axis=1).T


def save_compressed_phi_coords(phi_coords, filename):
    np.savez_compressed(filename, phi_coords=phi_coords)


def process_arcname(list_fn, base_dir):
    return [os.path.relpath(fn, start=base_dir) for fn in list_fn]


def load_gt_label(fn):
    assert os.path.exists(fn), f"Not found {fn}"
    return np.load(fn)["phi_coords"]


def print_cfg_information(cfg):
    logging.info(f"Experiment ID: {cfg.id_exp}")
    logging.info(f"Output_dir: {cfg.output_dir}")


def check_existence_in_list_fn(list_fn):
    check = [os.path.exists(fn) for fn in list_fn]
    if all(check):
        return True
    else:
        print(f"Missing files: {np.asarray(list_fn)[np.logical_not(check)]}")
        return False


def check_file_exist(list_files, ext):
    pool = ThreadPool(processes=10)
    list_threads = []
    for fn in list_files:
        list_threads.append(pool.apply_async(os.path.isfile, (f"{fn}.{ext}",)))
    local_data = []
    for thread in tqdm(list_threads):
        local_data.append(thread.get())
    return local_data


def get_abs_path(file):
    return os.path.dirname(os.path.abspath(file))
