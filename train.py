import yaml
import importlib
from importlib.machinery import SourceFileLoader
from argparse import ArgumentParser
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from debug_util import save_modules

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    pl.seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-d', '--description', type=str, help='description of the experiment', default='')
    parser.add_argument('-s', '--save_dir', type=str, default='checkpoints/nuSArgo')

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--wonly', action='store_true', default=False)

    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--save_top_k', type=int, default=-1)
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--viz_goalpred', action='store_true', default=False)
    parser.add_argument('--gpus', type=int, default=1)
    
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--monitor', type=str, default='val/ADE_T')

    args = parser.parse_args()

    with open(args.config, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
        cfg['model_specific']['kwargs']['viz'] = args.viz
        cfg['model_specific']['kwargs']['viz_goalpred'] = args.viz_goalpred
        cfg['description'] = args.description
        args.max_epochs = cfg['training_specific']['max_epochs']

    model = getattr(SourceFileLoader(cfg['model_specific']['module_name'], cfg['model_specific']['file_path']).load_module(cfg['model_specific']['module_name']), cfg['model_specific']['module_name'])
    model = model(**dict(cfg))

    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    logger = TensorBoardLogger(save_dir=args.save_dir, name=args.name)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint], logger=logger, num_sanity_val_steps = 0)

    dmodulecfg = cfg['datamodule_specific']
    datamodule = getattr(SourceFileLoader(dmodulecfg['module_name'], dmodulecfg['file_path']).load_module(dmodulecfg['module_name']), dmodulecfg['module_name'])
    datamodule = datamodule(**dict(dmodulecfg['kwargs']))

    save_modules(logger.log_dir, args.config, cfg)
    
    if args.wonly:
        model = model.load_from_checkpoint(checkpoint_path=args.ckpt)
        trainer.fit(model, datamodule)
    else:
        trainer.fit(model, datamodule, ckpt_path=args.ckpt)
    