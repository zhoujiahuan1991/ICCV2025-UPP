import os
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
import open3d as o3d
from utils import misc
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

@DATASETS.register_module()
class RealSensorDataset(data.Dataset):
    def __init__(self, config):
        """PyTorch Dataset Wrapper"""
        test_path = config.dataPath
        self.npoints = config.N_POINTS
        dirname = os.listdir(test_path)
        dirname = sorted(dirname)
        self.datapath = []
        for cate in dirname:
            newPath = os.path.join(test_path, cate)
            objects = os.listdir(newPath)
            objects = sorted(objects)
            ccounter = 0
            for obj in objects:
                ccounter += 1
                if ccounter > 50:
                    break
                obj_label = cate
                self.datapath.append([obj_label, os.path.join(newPath, obj)])

        self.catfile = os.path.join('data/ModelNet/modelnet40_normal_resampled', 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

    def __len__(self):
        return len(self.datapath)

    def pc_norm(self, partail_scannet):
        """pc: NxC, return NxC"""
        m = np.max(np.sqrt(np.sum(partail_scannet**2, axis=1))) * 2

        partail_scannet = partail_scannet / m
        partail_scannet = torch.from_numpy(partail_scannet)
        return partail_scannet

    def __getitem__(self, index):
        insta = self.datapath[index]
        partail_scannet = np.array(o3d.io.read_point_cloud(insta[1]).points)
        partail_scannet = self.pc_norm(partail_scannet)
        partail_scannet = np.array(partail_scannet, dtype=float)

        cls = self.classes[self.datapath[index][0]]
        label = np.array([cls]).astype(np.int32)
        current_points = torch.from_numpy(partail_scannet).float().cuda()
        current_points = misc.fps(current_points[None,...], self.npoints)[0][0]
        # return insta[0], partail_scannet, insta[1]
        return 'RealSensor', 'sample', (current_points, label)
