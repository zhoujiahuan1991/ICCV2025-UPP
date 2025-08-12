import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class ProjectedShapeNetDataset(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_root = config.COMPLETE_POINTS_ROOT
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS
        self.task = config.TASK
        self.n_renderings = config.N_RENDERINGS if self.subset == 'train' else 1
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'Projected_ShapeNet')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        missing_file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0].split('/')[-1]
            model_id = line.split('-')[1].split('.')[0]
            if os.path.exists(self.partial_points_path % (taxonomy_id, model_id, 0)):
                if config.CARS:
                    if taxonomy_id == '02958343':
                        self.file_list.append({
                            'taxonomy_id': taxonomy_id,
                            'model_id': model_id,
                            'file_path': line
                        })
                    else:
                        pass
                else:
                    self.file_list.append({
                        'taxonomy_id': taxonomy_id,
                        'model_id': model_id,
                        'file_path': line
                    })
            else:
                missing_file_list.append(line)
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
        # mapping taxonomy_id to 55 class labels
        label_set = sorted(list(set([item['taxonomy_id'] for item in self.file_list])))
        self.label_map = {}
        for i in range(len(label_set)):
            self.label_map[label_set[i]] = i

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        gt_path = os.path.join(self.complete_points_root, sample['file_path'])
        data['gt'] = IO.get(gt_path).astype(np.float32)

        partial_path = self.partial_points_path % (sample['taxonomy_id'], sample['model_id'], rand_idx)
        data['partial'] = IO.get(partial_path).astype(np.float32)
        
        assert data['gt'].shape[0] == self.npoints
        if self.task == 'completion':
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])
        elif self.task == 'classification':
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], self.label_map[sample['taxonomy_id']])
        else:
            raise NotImplementedError('Unspecific task for getting dataset item.')

    def __len__(self):
        return len(self.file_list)