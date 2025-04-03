import os
import geometry_perception_utils.config_utils
from pathlib import Path

HM3D_DATA_COLLECTOR_ROOT = os.path.dirname(os.path.abspath(__file__))
HM3D_DATA_COLLECTOR_CFG_DIR = os.path.join(HM3D_DATA_COLLECTOR_ROOT, 'config')

os.environ['HM3D_DATA_COLLECTOR_ROOT'] = HM3D_DATA_COLLECTOR_ROOT
os.environ['HM3D_DATA_COLLECTOR_CFG_DIR'] = HM3D_DATA_COLLECTOR_CFG_DIR
