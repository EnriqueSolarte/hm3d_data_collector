import numpy as np
"""
Implementation based on pyntcloud
https://github.com/daavoo/pyntcloud
"""

def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out

class DenseVoxelGrid:  

    def __repr__(self):
        default = dict(
            voxel_centers=self.voxel_centers.shape,
            shape=self.shape,
            sizes=self.sizes,
            xyz_max=self.xyz_max,
            xyz_min=self.xyz_min,
        )
        
        caption = [f"* {self.__class__.__name__}"]
        caption += [f"{k}: {v}" for k, v in default.items()]
        return "\n\t * ".join(caption)
    
    def __init__(self,
                 points,
                 grid_size=0.01, 
                 keep_points=False
                 ):
        """Grid of voxels.
        Parameters
        ----------
        points: (N, 3) numpy.array
        """        

        self.x_y_z = np.asarray([1, 1, 1])
        self.sizes = np.asarray([grid_size, grid_size, grid_size])
        
        self.xyz_min, self.xyz_max = None, None
        self.segments = None
        self.shape = None
        self.n_voxels = None
        self.voxel_x, self.voxel_y, self.voxel_z = None, None, None
        self.voxel_n = None
        self.voxel_centers = None
        # ! At the last so the setter is called correctly
        if keep_points:
            self.points = points
        else:
            self.compute(points)

    def compute(self, points):
        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)
        xyz_range = abs(xyz_max - xyz_min)

        # adjust to obtain a minimum bounding box with all sides of equal length
        margin = max(xyz_range) - xyz_range
        xyz_min = xyz_min - margin / 2
        xyz_max = xyz_max + margin / 2

        for n, size in enumerate(self.sizes): 
            margin = (((xyz_range[n] // size) + 1) * size) - xyz_range[n]
            xyz_min[n] -= margin / 2
            xyz_max[n] += margin / 2
            self.x_y_z[n] = ((xyz_max[n] - xyz_min[n]) / size).astype(int)

        self.xyz_min = xyz_min
        self.xyz_max = xyz_max

        segments = []
        shape = []
        for i in range(3):
            # note the +1 in num
            s, step = np.linspace(xyz_min[i],
                                  xyz_max[i],
                                  num=(self.x_y_z[i] + 1),
                                  retstep=True)
            segments.append(s)
            shape.append(step)

        self.segments = segments

        self.n_voxels = np.prod(self.x_y_z)
        
        self.id = "V({},{})".format(self.x_y_z, self.sizes)
        
        self.shape = tuple(self.x_y_z)
        # find where each point lies in corresponding segmented axis
        # -1 so index are 0-based; clip for edge cases
        self.voxel_x = np.clip(np.searchsorted(self.segments[0], points[:, 0]), 0, self.x_y_z[0]-1)
        self.voxel_y = np.clip(np.searchsorted(self.segments[1], points[:, 1]), 0, self.x_y_z[1]-1)
        self.voxel_z = np.clip(np.searchsorted(self.segments[2], points[:, 2]), 0,  self.x_y_z[2]-1)
        self.voxel_n = np.ravel_multi_index([self.voxel_x, self.voxel_y, self.voxel_z], self.x_y_z)

        # compute center of each voxel
        mid_segments = [(self.segments[i][1:] + self.segments[i][:-1]) / 2 for i in range(3)]
        self.voxel_centers = cartesian(mid_segments).astype(np.float32)
        
        
    def query_idx(self, points):
        """Query the voxel grid indexes given the passed points.
        """
        assert points.shape[1] == 3, "points.shape[1] != 3"
        
        voxel_x = np.clip(np.searchsorted(
            self.segments[0], points[:, 0]), 0, self.x_y_z[0]-1)
        voxel_y = np.clip(np.searchsorted(
            self.segments[1], points[:, 1]), 0, self.x_y_z[1]-1)
        voxel_z = np.clip(np.searchsorted(
            self.segments[2], points[:, 2]), 0, self.x_y_z[2]-1)
        voxel_n = np.ravel_multi_index([voxel_x, voxel_y, voxel_z], self.x_y_z)
        return voxel_n

    def query_centers(self, points):
        """Query the voxel grid centers given the passed points.
        """
        assert points.shape[1] == 3, "points.shape[1] != 3"
        
        indexes = self.query_idx(points)
        indexes = np.unique(indexes)
        return self.voxel_centers[indexes]
    
    
if __name__ == '__main__':
    import os 
    from pathlib import Path
    from geometry_perception_utils.vispy_utils import plot_list_pcl
    
    # * Load data sample (point cloud)
    root = os.path.dirname(__file__)
    point_cloud_path = Path(f'{root}/samples/bunnyStatue.txt').resolve().__str__()
    point_cloud = np.loadtxt(point_cloud_path, delimiter=' ')
    
    points = point_cloud[:,:3]
    colors = point_cloud[:,3:6]
    normals = point_cloud[:,6:]
    
    print("points.shape: ", points.shape)
    print("colors.shape: ", colors.shape)
    print("normals.shape: ", normals.shape)
    
    # * Create a voxel grid
    voxel_grid = DenseVoxelGrid(points)
    print("voxel_grid.id: ", voxel_grid.id)
    print("voxel_grid.shape: ", voxel_grid.shape)
    print(voxel_grid)

