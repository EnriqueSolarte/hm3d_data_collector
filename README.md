# HM3D Data Collector
#### This repository is the python implementation that: (1) accesses to habitat-sim, (2) allows the user to collect data, (3) renders data (different sensor modalities), and (4) post-processes the data.
![Main GIF](assets/main.gif)
---
## Installation
#### 1 Create conda env (highly recommended)
```bash
conda create -n hm3d_collector python=3.9 
conda activate hm3d_collector
```

#### 2. Clone and install the repository as a python library
```bash
git clone git@github.com:EnriqueSolarte/hm3d_data_collector.git

cd hm3d_data_collector

pip install -e . #for development
pip install . #for production
```

#### 3. Install habitat-sim
This is the most tedious part of the installation. If the following command does not work, please follow the instructions in the [habitat-sim repository](https://arc.net/l/quote/qolneuio). 

```bash
conda install habitat-sim -c conda-forge -c aihabitat
``` 

#### 4. Download habitat-sim assets (dataset) to render the scenes

This repository uses the [hm3d dataset](https://aihabitat.org/datasets/hm3d/) released in NeurIPS'21. For practical purposes within ITRI, we downloaded this data (for research purposes only) at:
```
/media/datasets/habitat/v0.2
```
Please contact me at for get access to this data. `enrique.solarte.pardo@itri.org.tw`

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
These format is used only to refer the dataset split name, scene name and idx of the scene. For instance. 
```yaml
hm_split: minival
hm_idx_scene: 00800
hm_scene_name: TEEsavR23oF 
```
This information can be manually put in the `./examples/cfg.yaml` file or passed by command line as follows:

```shell
python examples/collect_data.py hm_split=minival hm_idx_scene=00800 hm_scene_name=TEEsavR23oF
```