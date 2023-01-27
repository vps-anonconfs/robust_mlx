import os
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np

from src.module import (
    LitClassifier,
    LitIBPClassifier,
    LitRRRClassifier,
    LitCDEPClassifier
    LitPGDClassifier,
    LitCoRMClassifier,
)
import neptune.new as neptune

from src.configs.utils import populate_defaults
from src.configs.user_defaults import user_defaults
from src.datasets.data_module import DataModule


def cli_main():
    
    """
    Best hyperparameter can be found in configs/expts.py
    """
    # ------------ args -------------
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, help="Name of config expt from config.expts")
    parser.add_argument("--seed", default=1234, type=int, help="random seeds")
    parser.add_argument("--alg", type=str, default="erm", help="rrr or ibp or erm")
    parser.add_argument("--project", default="IBP2", type=str, help="a name of project to be used")
    parser.add_argument("--dataset", default="decoy_cifar10", type=str, help="dataset to be loaded")

    # data related
    # todo: data_seed is not used yet
    parser.add_argument("--data_seed", default=1234, type=int, help="batchsize of data loaders")
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
    parser.add_argument("--batch_size_train", default=16, type=int, help="batchsize of data loaders")
    parser.add_argument("--batch_size_test", default=30, type=int, help="batchsize of data loaders")
    parser.add_argument("--data_dir", type=str, help="directory of cifar10 dataset")
    parser.add_argument("--num_classes", type=int, help="Number of labels")
    parser.add_argument("--num_groups", type=int, help="Number of groups")
    parser.add_argument("--data_frac", type=float, default=1., help="Fraction of training data to use (for debugging)")
    parser.add_argument("--dataset_kwargs", default={},
                        help="Special dataset related kwargs that is passed to dataset constructor")

    # model related
    parser.add_argument("--learning_rate", type=float, help="learning rate of optimizer")
    # milestones for lr scheduling
    # todo: bad default for milestones?
    parser.add_argument("--milestones", nargs="+", default=[100, 150], type=int,
                        help="learning rate scheduler for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay of optimizer")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"], help="optimizer")
    parser.add_argument("--use-weighted-ce", type=bool, default=True, 
                        help="Use weighted ce?", action=argparse.BooleanOptionalAction)

    # network related
    parser.add_argument("--network_name", type=str,
                        help="name of the backbone network, should be a value supported by networks.get_network")
    parser.add_argument("--network_kwargs", help="network loading kwargs")
    parser.add_argument("--initialization-factor", type=float, default=None, help="Initialization factor of parameters")

    # alg related
    # --ibp
    parser.add_argument("--ibp_ALPHA", type=float, default=0.0, help="Regularization Parameter (Weights the Reg. Term)")
    parser.add_argument("--ibp_EPSILON", type=float, default=0.0, help="Input Perturbation Budget at Training Time")
    parser.add_argument("--ibp_start_EPSILON", type=float, default=0.0, help="Starting input perturbation")
    parser.add_argument("--ibp_rrr", type=float, default=0, help="rrr like loss wt in IBP")
    # --rrr
    parser.add_argument("--rrr_ap_lamb", type=float, default=0.0)
    # --heatmaps args for rrr
    parser.add_argument("--rrr_hm_method", type=str, default="rrr", help="interpretation method")
    parser.add_argument("--rrr_hm_norm", type=str, default="none")
    parser.add_argument("--rrr_hm_thres", type=str, default="none")
    # --cdep
    parser.add_argument("--cdep_ap_lamb", type=float, default=0.0)
    # --corm
    parser.add_argument("--corm_EPSILON", type=float, default=0.0)
    

    # neptune related config
    parser.add_argument("--user", type=str, default='vihari',
                        help=f"Must be one of {user_defaults.keys()} for setting user related config.")
    parser.add_argument("--api_key", type=str, default=None, help="Neptune api key to upload logs")
    parser.add_argument("--ID", type=str, default=None, help="Neptune ID to upload logs")
    parser.add_argument("--cache-fldr", type=str, help="Folder to save logs")

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    populate_defaults(args)
    args.default_root_dir = f"{args.cache_fldr}/{args.project}"
    print("Args:", vars(args))

    # Trainer
    if args.alg.lower() == "erm":
        classifier = LitClassifier
    elif args.alg.lower() == "rrr":
        classifier = LitRRRClassifier
    elif args.alg.lower() == "ibp":
        classifier = LitIBPClassifier
    elif args.alg.lower() == 'cdep':
        classifier = LitCDEPClassifier
    elif args.alg.lower() == 'pgd':
        classifier = LitPGDClassifier
    elif args.alg.lower() == 'corm':
        classifier = LitCoRMClassifier
    else:
        raise Exception("regularizer name error")

    pl.seed_everything(args.seed)
    # ------------ data -------------
    data_module = DataModule(**vars(args))
    print(f"Training on dataset of size {len(data_module.train_dataset)}, "
          f"val and test size {len(data_module.val_dataset)}, {len(data_module.test_dataset)}")

    # ------------ logger -------------
    run = neptune.init_run(
        # api_token=args.api_key,
        # project=f"{args.ID}/{args.project}",
        capture_stdout=False,
        mode="debug"
    )
    logger = NeptuneLogger(run=run, log_model_checkpoints=False)
    dirpath = os.path.join(args.default_root_dir, logger.version)

    # ------------ callbacks -------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        monitor="valid_acc_wg",
        filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
        save_last=True,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    # ------------ model -------------
    model_kwargs = vars(args)
    if args.use_weighted_ce:
        class_weights = torch.tensor(data_module.get_class_weights(), dtype=torch.float32)
    else:
        class_weights = None
    print("Class weights:", class_weights)
    model_kwargs["class_weights"] = class_weights
    model = classifier(**model_kwargs)

    run["parameters"] = model_kwargs
    run["sys/tags"].add([args.name, f"seed:{args.seed}"])
    # ------------ run -------------
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, dataloaders=data_module)
    run.stop()


if __name__ == "__main__":
    cli_main()
