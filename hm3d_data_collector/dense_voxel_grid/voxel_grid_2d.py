import dvf_map
import hydra
from geometry_perception_utils.io_utils import get_abs_path
from geometry_perception_utils.vispy_utils import plot_list_pcl
import numpy as np
import time
from omegaconf import OmegaConf

class XYZData:
    def __init__(self, xyz, data):
        self.data=[data]
        self.xyz = xyz
        
class BEVMap:
    def __init__(self):
        self.list_xyz_data = []
    
    def add_data(self, xyz, data):
        xyz_data = self.get_xyz_data(xyz)
        if xyz_data is None:
            self.list_xyz_data.append(XYZData(xyz, data))
        else: 
            if data not in xyz_data.data:
                xyz_data.data.append(data)
                                    
    def get_xyz_data(self, xyz) -> XYZData:
        if self.list_xyz_data == []:
            return None
        idx = [np.sum(dt.xyz - xyz) == 0 for dt in self.list_xyz_data]
        if np.sum(idx) == 0:
            return None
        return self.list_xyz_data[np.argmin(idx)]
    
class VoxelGrid2D:
    @property
    def u_bins(self):
        return self._u_bins

    @u_bins.setter
    def u_bins(self, value):
        self._u_bins = value
        self.w = self._u_bins.shape[0]
    
    @property
    def v_bins(self):
        return self._v_bins

    @v_bins.setter
    def v_bins(self, value):
        self._v_bins = value
        self.h = self._v_bins.shape[0]
          
    @property
    def shape(self):
        return (self.h, self.w)
       
    def __init__(self, cfg):
        assert cfg.voxel_type == 'voxel_grid_2d'
        [setattr(self, k, v) for k, v in cfg.items()]

        number_grids = int(1/self.grid_size) 
        self.u_bins = np.linspace(0, 1 - self.grid_size, number_grids)
        self.v_bins = np.linspace(0, 1 - self.grid_size, number_grids)
        
    def get_bins(self):
        return self.u_bins, self.v_bins
    
    @classmethod
    def from_bins(clc, u_bins, v_bins):
        dict_ = OmegaConf.create(
            {'voxel_type': 'voxel_grid_2d', 
             'grid_size': float(u_bins[1] - u_bins[0]),
             })
        clc = VoxelGrid2D(dict_)
        clc.set_bins(u_bins, v_bins)
        return clc
        
    
    def set_bins(self, u_bins, v_bins):
        self.u_bins = u_bins
        self.v_bins = v_bins
    
    def extend_bins(self, points, bins):
        p_max = points.max()
        p_min = points.min()
        if p_max > bins[-1]:
            # add bins to the right
            exceed = int((p_max - bins[-1])/self.grid_size) + self.padding
            exceed_bins = [bins[-1] + self.grid_size * i for i in range(1, exceed)]
            bins = np.concatenate([bins, exceed_bins])
            
        if p_min < bins[0]:
            # add bins to the left
            exceed = int((abs(p_min) - abs(bins[0]))/self.grid_size) + self.padding
            exceed_bins = [bins[0] - self.grid_size * i for i in range(exceed, 0, -1)]
            bins = np.concatenate([exceed_bins, bins])
        return bins
    
    def get_uv_from_xyz(self, xyz):
        if xyz.size == 0:
            return None, None, None
        # u_bins --> z coord
        self.u_bins = self.extend_bins(xyz[2], self.u_bins)
        u = np.searchsorted(self.u_bins, xyz[2])
        
        # v_bins --> x coord
        self.v_bins = self.extend_bins(xyz[0], self.v_bins)
        v = np.searchsorted(self.v_bins, xyz[0])
        return np.vstack((u, v))
         
    def project_xyz(self, xyz):
        """
        Projects the xyz points into the voxel grid. 
        \n
        Returns  [xyz_vx, xyz_idx, vxl_idx]
        xyz_vx: The voxel centers
        xyz_idx: xyz indexes for the xyz_vx points
        vxl_idx: xyz_vx indexes for the xyz points
        """
        if xyz.size == 0:
            return None, None, None
        # u_bins --> z coord
        self.u_bins = self.extend_bins(xyz[2], self.u_bins)
        u = np.searchsorted(self.u_bins, xyz[2])
        
        # v_bins --> x coord
        self.v_bins = self.extend_bins(xyz[0], self.v_bins)
        v = np.searchsorted(self.v_bins, xyz[0])
        
        # Getting the index of the voxel
        vxl_idx = u*self.h + v # along v first and then u
        unique_idx, xyz_idx, vxl_idx = np.unique(vxl_idx, return_index=True, return_inverse=True)
        
        # getting voxels centers
        x_ = self.v_bins[unique_idx % self.h]+self.grid_size/2
        z_ = self.u_bins[unique_idx // self.h]+self.grid_size/2
        
        return np.vstack([x_, np.zeros(x_.shape), z_]), unique_idx, xyz_idx, vxl_idx
        
    def get_centroids_by_idx(self, idx):
        """
        Given a set of indexes, returns the voxel centers
        """
        # getting voxels centers
        z_ = self.u_bins[idx // self.h]+self.grid_size/2
        x_ = self.v_bins[idx % self.h]+self.grid_size/2
        return np.vstack([x_, np.zeros(x_.shape), z_])    
    
@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    
    voxel = VoxelGrid2D(cfg.voxel_grid_2d)
    pos = np.zeros((3, 1))
    for _ in range(10000):
        xyz = np.random.uniform(0, 1, (3, 1000)) + pos.reshape(3, 1)
        tic = time.time()
        xyz_, idx = voxel.project_xyz(xyz)
        t = time.time()-tic
        print(xyz.shape, xyz_.shape, voxel.shape, t)
        # plot_list_pcl([xyz, xyz_], size=10)
        pos = xyz[:, -1]
        
        
        
if __name__ == '__main__':
    main()