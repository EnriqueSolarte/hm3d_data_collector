# HM3D Data Collector

> **Note:** This repository is the python implementation that: (1) accesses to habitat-sim, (2) allows the user to collect data, and (3) renders data (different sensor modalities).

![Main GIF](assets/main.gif)
---
## Installation
#### 1 Create conda env (highly recommended)
```bash
conda create -n hm3d_collector python=3.9 
conda activate hm3d_collector
```

#### 2. Install habitat-sim
This is the most tedious part of the installation. 
```bash
conda install habitat-sim -c conda-forge -c aihabitat
``` 
If the above command does not work, try to install from source 

```bash
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install
```
If the above command does not work either , please check [habitat-sim repository](https://arc.net/l/quote/qolneuio) for help. 


#### 3. Clone and install the repository as a python library
```bash
git clone git@github.com:EnriqueSolarte/hm3d_data_collector.git

cd hm3d_data_collector

pip install -e . #for development
pip install . #for production
```
> 🚀 **Tip:** You can install this package using pip.
```bash
pip install git+https://github.com/EnriqueSolarte/hm3d_data_collector.git@latest
```

Install the geometry perception utils package. 
```bash
pip install git+https://github.com/EnriqueSolarte/geometry_perception_utils.git@latest
```

#### 4. Download habitat-sim assets (dataset) to render the scenes

This repository uses the [hm3d dataset](https://aihabitat.org/datasets/hm3d/) released in NeurIPS'21. For practical purposes, you can assume that the data is stored at:
```
/media/datasets/habitat/v0.2
```
in the cofig file `./examples/cfg.yaml` you can set the path to the dataset. 

```yaml
hm3d_dir: /media/datasets/habitat/v0.2
```

## How to use it

All the scenes that content semantic data are listed at `./examples/list_scenes.json`. Example:
```json
{   ...
	"minival": {
		"00800-TEEsavR23oF": "00800-TEEsavR23oF/TEEsavR23oF.semantic.glb",
		"00802-wcojb4TFT35": "00802-wcojb4TFT35/wcojb4TFT35.semantic.glb",
		"00803-k1cupFYWXJ6": "00803-k1cupFYWXJ6/k1cupFYWXJ6.semantic.glb",
		"00808-y9hTuugGdiq": "00808-y9hTuugGdiq/y9hTuugGdiq.semantic.glb"
	}
    ...
}
```
This format is used only to refer the dataset split name, scene name and idx of the scene. For instance. 
```yaml
hm_split: minival
hm_idx_scene: 00800
hm_scene_name: TEEsavR23oF 
```
This information can be manually put in the `./examples/cfg.yaml` file or passed by command line as follows:

```shell
python examples/collect_scene.py hm_split=minival hm_idx_scene=00800 hm_scene_name=TEEsavR23oF
```

### Data versioning

All scenes can be collected multiple times. Maybe, you want to collect different rooms or use a different path each time. To keep track of these versions, you can use the `hm_data_version` parameter in `./examples/cfg.yaml`. For instance, if you want to collect a scene with version `1` use the following command:

```shell
python examples/collect_scene.py hm_split=minival hm_idx_scene=00800 hm_scene_name=TEEsavR23oF hm_data_version=1
```
We can also change the `hm_data_version` parameter manually in the `./examples/cfg.yaml` file.

```yaml
...
hm_split: minival
hm_idx_scene: 00800
hm_scene_name: TEEsavR23oF 
hm_data_version: 1
...
```
The directory for the collected data is defined by the `data_dir` parameter. The path to the raw data (HM3D) is defined in `hm3d_dir`. Both parameter can been set in the `./examples/cfg.yaml` file or passed as arguments in CLI.

### Run scripts
After setting the right parameters for a particular scene and version, you can run:

```shell
python collect_scene.py
``` 
Then, to render the scene, you can run:

```shell
python render_scene.py
```
To visualize the scene in 3D, you can run:

```shell
python visualize_scene.py
```

![scene GIF](assets/scene_3d.gif)


