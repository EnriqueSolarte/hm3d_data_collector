import logging
import os
import git
import yaml
from omegaconf import OmegaConf, open_dict
from coolname import generate_slug
import datetime
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
import importlib
import shutil
from hm3d_data_collector.utils.io_utils import create_directory
import numpy as np


def set_stamp_name(number_names, *, _parent_):
    _parent_['stamp_name'] = generate_slug(int(number_names))
    return _parent_['stamp_name']


def get_hydra_file_dirname(*, _parent_):
    return HydraConfig.get().runtime.config_sources[1].path


def get_hydra_dirname(*, _parent_):
    return HydraConfig.get().runtime.config_sources[1].path.split("/")[-1]


def get_date(format=0, *, _parent_):
    if format == 0:
        return datetime.datetime.now().strftime("%b %d '%y %H:%M:%S")
    elif format == 1:
        return datetime.datetime.now().strftime("%b-%Y")
    elif format == 2:
        return datetime.datetime.now().strftime("%d-%b-%Y")
    elif format == 3:
        return datetime.datetime.now().strftime("%y%m%d.%H%M%S")
    elif format == 4:
        return datetime.datetime.now().strftime("%y_%m_%d")
    else:
        return datetime.datetime.now().strftime("%Y-%m-%d")


def get_hostname(*, _parent_):
    return os.uname().nodename


def get_git_commit(*, _parent_):
    repo = git.Repo(search_parent_directories=True)
    git_commit = repo.head._get_commit().name_rev
    return git_commit


def get_timestamp(format, *, _parent_):
    _parent_['stamp'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return _parent_['stamp']


def load(input_file, *, _parent_):
    assert os.path.exists(input_file), f"File does not exist {input_file}"
    cfg = read_omega_cfg(input_file)
    return cfg


def get_hydra_overrides(**args):
    overrides = HydraConfig.get()['overrides'].get('task', [])
    overrides = [o for o in overrides if 'cuda' not in o]
    all_args = "__".join(overrides)
    all_args = all_args.replace("=", "_")
    return all_args


def get_dependency_versions(module, *, _parent_):
    func = getattr(importlib.import_module(
        module, package=None), 'get_dependencies')
    return func()


OmegaConf.register_new_resolver('set_stamp_name', set_stamp_name)
OmegaConf.register_new_resolver('get_hostname', get_hostname)
OmegaConf.register_new_resolver('get_git_commit', get_git_commit)
OmegaConf.register_new_resolver('get_timestamp', get_timestamp)
OmegaConf.register_new_resolver('get_date', get_date)
OmegaConf.register_new_resolver('get_hydra_overrides', get_hydra_overrides)
OmegaConf.register_new_resolver(
    'get_hydra_file_dirname', get_hydra_file_dirname)
OmegaConf.register_new_resolver(
    'get_dependency_versions', get_dependency_versions)
OmegaConf.register_new_resolver('load', load)
OmegaConf.register_new_resolver('get_hydra_dirname', get_hydra_dirname)


def get_empty_cfg():
    logging.basicConfig(format='[%(levelname)s] [%(asctime)s]:  %(message)s',
                        level=logging.INFO)
    cfg_dict = dict()

    # ! add git commit
    # cfg_dict = add_git_commit(cfg_dict)

    cfg = OmegaConf.create(cfg_dict)
    return cfg


def save_cfg(cfg, cfg_file=None, save_list_scripts=None):
    """
    Automatically saves the cfg files in the log_dir. Additionally, it present 
    the option to save the script and other scripts that are used.
    Args:
        cfg: General CFG where the cfg.log_dir is defined.
        script (optional): script filename that want to be saved.
        cfg_file (optional): cfg yaml filename. By default the passed cfg will be used
        save_list_scripts (optional): list of other filenames scripts that we want to saved.
    """

    create_directory(cfg.log_dir, delete_prev=False)
    if cfg_file is None:
        cfg_file = os.path.join(cfg.log_dir, "cfg.yaml")

    if save_list_scripts is not None:
        for s in save_list_scripts:
            try:
                dest_fn = os.path.join(
                    cfg.log_dir, os.path.basename(s))
                shutil.copy(s, dest_fn)
            except:
                logging.warning(f"Could not copy script {s}")

    try:
        # OmegaConf.resolve(cfg)
        cfg.date = cfg.date
        cfg.time = cfg.time
        with open(cfg_file, 'w') as fn:
            OmegaConf.save(config=cfg, f=fn)
        logging.info(f"Saved cfg to {cfg_file}")
    except:
        logging.warning(f"Could not save cfg to {cfg_file}")


def read_omega_cfg(cfg_file):
    assert os.path.exists(cfg_file), f"File does not exist {cfg_file}"

    # ! Reading YAML file
    with open(cfg_file, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = OmegaConf.create(cfg_dict)
    return cfg


def read_cfg(cfg_file):
    cfg = read_omega_cfg(cfg_file)
    OmegaConf.resolve(cfg)
    return cfg


def merge_cfg(list_cfg):
    cfg = {}
    for c in list_cfg:
        cfg = {**cfg, **c}
    return OmegaConf.merge(cfg, cfg)


# ! Registering current git commit
def get_repo_version(REPO_DIR):
    current_dir = os.getcwd()
    if os.path.isdir(REPO_DIR):
        os.chdir(REPO_DIR)
    else:
        os.chdir(os.path.dirname(REPO_DIR))

    repo = git.Repo(search_parent_directories=True)
    commit = repo.head._get_commit()
    repo_name = Path(repo.working_dir).stem
    data = f"{repo_name}: {commit.name_rev}"
    os.chdir(current_dir)
    return data


def update_cfg(cfg, new_struct):
    with open_dict(cfg):
        cfg.update(new_struct)
    return cfg


def get_hydra_log_dir():
    log_dir = HydraConfig.get().runtime.output_dir
    return log_dir


def print_log_dir():
    log_dir = get_hydra_log_dir()
    print(f" >>>>  Current log dir:")
    print(f" >>>>  {log_dir}")


def print_file(fn):
    with open(fn, 'r') as f:
        script_contents = f.read()
        print(script_contents)


if __name__ == '__main__':
    test = get_repo_version(__file__)
    print(test)
