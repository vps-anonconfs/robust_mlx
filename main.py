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
)
import neptune.new as neptune

from src.configs.utils import populate_defaults
from src.configs.user_defaults import user_defaults
from src.datasets.data_module import DataModule


def cli_main():
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
    parser.add_argument("--rrr_hm_thres", type=str, default="abs")
    # --cdep
    parser.add_argument("--cdep_ap_lamb", type=float, default=0.0)

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
    # logs = {
    #     'dcifar2': {'cdep': [], 'ibp_rrr': [991, 1003, 942], 'erm': [943, 948], 'rrr': [1001, 992, 945],
    #                 'ibp': [1039, 1031, 1013]},
    #     'isic': {'cdep': [359, 355, 353], 'rrr': [], 'erm': [339, 343, 334], 'ibp': [335, 336, 338]},
    #     'plant': {'cdep': [695, 696, 697], 'erm': [683, 684, 685], 'ibp': [687, 689, 690]}
    # }
    # dset = 'dcifar'
    # for alg in logs[dset].keys(): # , "IBP2-682"]: # ["IBP2-687", "IBP2-689", "IBP2-690"]: # ["IBP2-683", "IBP2-684", "IBP2-685"]:
    #         # ["IBP2-611", "IBP2-606", "IBP2-584", "IBP2-598", "IBP2-581"]:
    #     #["IBP2-359", "IBP2-355", "IBP2-353"]: #["IBP2-335", "IBP2-336", "IBP2-338"]: # ["IBP2-334", "IBP2-339", "IBP2-343"]: # ["IBP2-345", "IBP2-346", "IBP2-344"]:
    #     all_test_stats = []
    #     for run in logs[dset][alg]:
    #         fldr = f"{args.default_root_dir}/IBP2-{run}"
    #         fname = [fname for fname in os.listdir(fldr) if fname.startswith('checkpt')][0]
    #         # fname = "last.ckpt"
    #         fldr = f"{fldr}/{fname}"
    #         test_stats = trainer.test(model, dataloaders=data_module, ckpt_path=fldr)[0]
    #         all_test_stats.append(test_stats)
    #     if len(all_test_stats) > 0:
    #         print(f"Alg: {alg}")
    #         print("----------")
    #         for metric in test_stats.keys():
    #             mvals = [ts[metric] for ts in all_test_stats]
    #             print(f"{metric}: {np.mean(mvals), np.std(mvals)}")
    #
    # print(len(data_module.test_dataset), type(data_module.test_dataset), len(data_module.test_dataloader()))
    # for dl in data_module.test_dataloader():
    #     trainer.test(model, dataloaders=dl,
    #                  # ckpt_path=f"{args.default_root_dir}/IBP2-260/last.ckpt"
    #                  # ckpt_path=f"{args.default_root_dir}/IBP2-261/checkpt-epoch=09-valid_acc=0.71.ckpt"
    #                  ckpt_path=f"{args.default_root_dir}/IBP2-274/last.ckpt"
    #                  )
    run.stop()


if __name__ == "__main__":
    cli_main()
