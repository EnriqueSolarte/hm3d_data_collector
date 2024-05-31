import dvf_map
import hydra
from hm3d_data_collector.utils.io_utils import get_abs_path
from hm3d_data_collector.utils.vispy_utils import plot_list_pcl
import numpy as np
import time
from omegaconf import OmegaConf


class VoxelGrid3D:
    @property
    def c_bins(self):
        return self._c_bins

    @c_bins.setter
    def c_bins(self, value):
        self._c_bins = value
        self.c = self._c_bins.shape[0]

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
        return (self.h, self.w, self.c)

    def get_bins(self):
        return self.u_bins, self.v_bins, self.c_bins

    def set_bins(self, u_bins, v_bins, c_bins):
        self.u_bins = u_bins
        self.v_bins = v_bins
        self.c_bins = c_bins

    def __init__(self, cfg):
        assert cfg.voxel_type == 'voxel_grid_3d'
        [setattr(self, k, v) for k, v in cfg.items()]

        number_grids = int(1/self.grid_size)
        self.u_bins = np.linspace(0, 1 - self.grid_size, number_grids)
        self.v_bins = np.linspace(0, 1 - self.grid_size, number_grids)
        self.c_bins = np.linspace(0, 1 - self.grid_size, number_grids)

    @classmethod
    def from_bins(clc, u_bins, v_bins, c_bins):
        dict_ = OmegaConf.create(
            {'voxel_type': 'voxel_grid_3d',
             'grid_size': float(u_bins[1] - u_bins[0]),
             })
        clc = VoxelGrid3D(dict_)
        clc.set_bins(u_bins, v_bins, c_bins)
        clc.grid_size = u_bins[1] - u_bins[0]
        return clc

    def extend_bins(self, points, bins):
        p_max = points.max()
        p_min = points.min()
        if p_max > bins[-1]:
            # add bins to the right
            exceed = int((p_max - bins[-1])/self.grid_size) + self.padding
            exceed_bins = [bins[-1] + self.grid_size *
                           i for i in range(1, exceed)]
            bins = np.concatenate([bins, exceed_bins])

        if p_min < bins[0]:
            # add bins to the left
            exceed = int(
                (abs(p_min) - abs(bins[0]))/self.grid_size) + self.padding
            exceed_bins = [bins[0] - self.grid_size *
                           i for i in range(exceed, 0, -1)]
            bins = np.concatenate([exceed_bins, bins])
        return bins

    def project_xyz(self, xyz):
        """
        Projects the xyz points into the voxel grid. [3 x n] 
        \n
        Returns  [xyz_vx, xyz_idx, vxl_idx]
        xyz_vx: The voxel centers [3 x v]
        xyz_idx: Indexes to reference the xyz points to a unique voxel [1 x v]
        vxl_idx: Indexes to reference the xyz points [1 x n]
        """
        if xyz.size == 0:
            return None, None, None

        # v_bins --> x coord
        self.v_bins = self.extend_bins(xyz[0], self.v_bins)
        v = np.searchsorted(self.v_bins, xyz[0])

        # w_bins --> y coord
        self.c_bins = self.extend_bins(xyz[1], self.c_bins)
        c = np.searchsorted(self.c_bins, xyz[1])

        # u_bins --> z coord
        self.u_bins = self.extend_bins(xyz[2], self.u_bins)
        u = np.searchsorted(self.u_bins, xyz[2])

        # Getting the index of the voxel

        vxl_idx = c * self.h * self.w + u*self.h + v  # along v first and then u
        unique_idx, xyz_idx, vxl_idx = np.unique(
            vxl_idx, return_index=True, return_inverse=True,)

        # getting voxels centers
        y_ = self.c_bins[unique_idx // (self.h * self.w)]+self.grid_size/2
        z_ = self.u_bins[(unique_idx % (self.h * self.w)) //
                         self.h]+self.grid_size/2
        x_ = self.v_bins[(unique_idx % (self.h * self.w)) %
                         self.h]+self.grid_size/2

        return np.vstack([x_, y_, z_]), unique_idx, xyz_idx, vxl_idx

    def get_centroids_by_idx(self, idx):
        """
        Given a set of indexes, returns the voxel centers
        """
        # getting voxels centers
        y_ = self.c_bins[idx // (self.h * self.w)]+self.grid_size/2
        z_ = self.u_bins[(idx % (self.h * self.w)) // self.h]+self.grid_size/2
        x_ = self.v_bins[(idx % (self.h * self.w)) % self.h]+self.grid_size/2
        return np.vstack([x_, y_, z_])


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):

    voxel = VoxelGrid3D(cfg.voxel_grid_3d)
    pos = np.zeros((3, 1))
    for _ in range(10000):
        xyz = np.random.uniform(-1, 1, (3, 10)) + pos.reshape(3, 1)
        tic = time.time()
        xyz_, xyz_idx, vxl_idx = voxel.project_xyz(xyz)
        t = time.time()-tic
        print(xyz.shape, xyz_.shape, voxel.shape, t)
        plot_list_pcl([xyz, xyz_], size=10)
        pos = xyz[:, -1]


if __name__ == '__main__':
    main()
