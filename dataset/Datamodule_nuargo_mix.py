from typing import Callable, Optional
import importlib
from importlib.machinery import SourceFileLoader

from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader


SPLIT_NAME = {'nuScenes': {'train': 'train', 'val': 'train_val', 'test': 'val', 'mini_train': 'mini_train', 'mini_val': 'mini_val'},
                'Argoverse': {'train': 'train', 'val': 'val', 'test': 'test_obs', 'sample': 'forecasting_sample'}}



class DataModuleNuArgoMix(LightningDataModule):

    def __init__(self,
                 dataset_file_path,
                 dataset_module_name,
                 **kwargs) -> None:
        super(DataModuleNuArgoMix, self).__init__()
        for k,v in kwargs.items():
            self.__setattr__(k, v)

        self.dataset_module = getattr(SourceFileLoader(dataset_module_name, dataset_file_path).load_module(dataset_module_name), dataset_module_name)

    def setup(self, stage: Optional[str] = None) -> None:

        self.train_dataset = self.dataset_module('train', self.nu_root, self.Argo_root, self.nu_dir, self.Argo_dir, spec_args=self.tr_dataset_args)
        self.val_dataset = self.dataset_module('val', self.nu_root, self.Argo_root, self.nu_dir, self.Argo_dir, spec_args=self.val_dataset_args)
        # self.test_dataset = self.dataset_module('test', self.nu_root, self.Argo2_root, self.nu_dir, self.Argo2_dir, spec_args=self.test_dataset_args)
        self.test_dataset = self.dataset_module('val', self.nu_root, self.Argo_root, self.nu_dir, self.Argo_dir, spec_args=self.test_dataset_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

if __name__ == '__main__':
    import yaml
    import importlib

    with open('/home/user/ssd4tb/frm_lightning/configs/hivt_LanesegGoal.yml', 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    dataconfig = cfg['datamodule_specific']
    datamodule = getattr(importlib.import_module(dataconfig['file_path']), dataconfig['module_name'])
    datamodule = datamodule(**dict(dataconfig['kwargs']))
    datamodule.setup()
    
    for idx, batch in enumerate(datamodule.train_dataloader()):
        print(batch)
        if idx>10:
            break
