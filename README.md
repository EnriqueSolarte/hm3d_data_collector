# HM3D Data Collector
#### This repository is the python implementation that: (1) access to habitat-sim, (2) allows the user to collect data, (3) render data (different sensor modalities), and (4) post-process the data.
![Main GIF](assets/main.gif)
---
## Installation
### 1 Create conda env (highly recommended)
```bash
conda create -n hm3d_collector python=3.9 
conda activate hm3d_collector
```

### 2. Clone the repository and install it as python library
```bash
git clone git@github.com:EnriqueSolarte/hm3d_data_collector.git

cd hm3d_data_collector

pip install -e . #for development
pip install . #for production
```

### 2. Install habitat-sim
This is the most tedious part of the installation. Please follow the instructions in the [habitat-sim repository](https://arc.net/l/quote/qolneuio). 

```bash
conda install habitat-sim -c conda-forge -c aihabitat
``` 

## How to use it