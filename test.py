import os

import yaml
from importlib.machinery import SourceFileLoader
from argparse import ArgumentParser

import pytorch_lightning as pl

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    pl.seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--viz', action="store_true", default=False)
    parser.add_argument('--viz_goalpred', action="store_true", default=False)
    parser.add_argument('--submit', action="store_true", default=False)
    parser.add_argument('--ood', action="store_true", default=False)
    parser.add_argument('--viz_ood', action="store_true", default=False)

    parser.add_argument('-s', '--save_dir', type=str, default='checkpoints')
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save_top_k', type=int, default=-1)
    parser.add_argument('--gpus', type=int, default=1)

    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])

    args = parser.parse_args()

    with open(args.config, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
        cfg['model_specific']['kwargs']['viz'] = args.viz
        cfg['model_specific']['kwargs']['viz_goalpred'] = args.viz_goalpred
        cfg['model_specific']['kwargs']['submit'] = args.submit
        cfg['model_specific']['kwargs']['ood'] = args.ood
        cfg['model_specific']['kwargs']['viz_ood'] = args.viz_ood

    model = getattr(SourceFileLoader(cfg['model_specific']['module_name'], cfg['model_specific']['file_path']).load_module(cfg['model_specific']['module_name']), cfg['model_specific']['module_name'])
    model = model(**dict(cfg))
    
    trainer = pl.Trainer.from_argparse_args(args, limit_test_batches=1.)
    trainer.logger = False

    dmodulecfg = cfg['datamodule_specific']
    datamodule = getattr(SourceFileLoader(dmodulecfg['module_name'], dmodulecfg['file_path']).load_module(dmodulecfg['module_name']), dmodulecfg['module_name'])
    datamodule = datamodule(**dict(dmodulecfg['kwargs']))

    trainer.test(model, dataloaders=datamodule, ckpt_path=args.ckpt)
    