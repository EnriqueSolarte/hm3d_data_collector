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

#### 2. Clone the repository and install it as python library
```bash
git clone git@github.com:EnriqueSolarte/hm3d_data_collector.git

cd hm3d_data_collector

pip install -e . #for development
pip install . #for production
```

#### 3. Install habitat-sim
This is the most tedious part of the installation. Please follow the instructions in the [habitat-sim repository](https://arc.net/l/quote/qolneuio). 

```bash
conda install habitat-sim -c conda-forge -c aihabitat
``` 

#### 4. Download habitat-sim assets (dataset) to render the scenes

This repository uses the [hm3d dataset](https://aihabitat.org/datasets/hm3d/) released in NeurIPS'21. For practical purposes within ITRI, we downloaded this data (for research purposes only) at:
```
/media/datasets/habitat/v0.2
```
Please contact me at for request access to this data. `enrique.solarte.pardo@itri.org.tw`

## How to use it