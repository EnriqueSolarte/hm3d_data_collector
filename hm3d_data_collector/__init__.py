import os
import geometry_perception_utils.config_utils
from pathlib import Path

HM_MAP_COLLECTOR_ROOT = os.path.dirname(os.path.abspath(__file__))
HM_MAP_COLLECTOR_CFG_DIR = os.path.join(HM_MAP_COLLECTOR_ROOT, 'config')
HM_MAP_COLLECTOR_CKPT_DIR = os.path.join(HM_MAP_COLLECTOR_ROOT, 'ckpt')

os.environ['HM_MAP_COLLECTOR_ROOT'] = HM_MAP_COLLECTOR_ROOT
os.environ['HM_MAP_COLLECTOR_CFG_DIR'] = HM_MAP_COLLECTOR_CFG_DIR
os.environ['HM_MAP_COLLECTOR_CKPT_DIR'] = HM_MAP_COLLECTOR_CKPT_DIR