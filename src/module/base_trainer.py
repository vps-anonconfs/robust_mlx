import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import torchmetrics

from src.module.utils.IBP_conv_functions import BoundSequential, AttributionIBPRegularizer
from src.networks import get_network


def populate_metrics(cls, num_classes, num_groups):
    # cls.train_auroc_metric = torchmetrics.AUROC(num_classes=num_classes)
    # cls.train_micro_accuracy = torchmetrics.Accuracy(num_classes=num_classes, average='micro')
    # cls.train_macro_accuracy = torchmetrics.Accuracy(num_classes=num_classes, average='macro')
    # cls.train_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro')
    # cls.valid_auroc_metric = torchmetrics.AUROC(num_classes=num_classes)
    # cls.valid_micro_accuracy = torchmetrics.Accuracy(num_classes=num_classes, average='micro')
    # cls.valid_macro_accuracy = torchmetrics.Accuracy(num_classes=num_classes, average='macro')
    # cls.valid_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro')
    # cls.train_metrics = {'auroc': cls.train_auroc_metric, 'mi_acc': cls.train_micro_accuracy,
    #                      'acc': cls.train_macro_accuracy, 'f1': cls.train_f1}
    # cls.valid_metrics = {'auroc': cls.valid_auroc_metric, 'mi_acc': cls.valid_micro_accuracy,
    #                      'acc': cls.valid_macro_accuracy, 'f1': cls.valid_f1}

    cls.train_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro')
    cls.valid_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro')

    cls.train_acc_per_group = torch.nn.ModuleList([torchmetrics.Accuracy(num_classes=num_classes, average='micro')
                                                   for _ in range(num_groups)])
    cls.valid_acc_per_group = torch.nn.ModuleList([torchmetrics.Accuracy(num_classes=num_classes, average='micro')
                                                   for _ in range(num_groups)])
    cls.train_metrics = {f'acc_g{gi}': cls.train_acc_per_group[gi] for gi in range(num_groups)}
    cls.valid_metrics = {f'acc_g{gi}': cls.valid_acc_per_group[gi] for gi in range(num_groups)}
    cls.train_metrics['acc'], cls.valid_metrics['acc'] = cls.train_acc, cls.valid_acc


def update_metrics(metrics, probs, y, g, num_groups):
    for gi in range(num_groups):
        this_idxs = torch.where(g == gi)[0]
        if len(this_idxs) > 0:
            metrics[f'acc_g{gi}'].update(probs[this_idxs], y[this_idxs])
    metrics['acc'].update(probs, y)


def get_metric_vals(metrics, mode):
    m_vals = dict([(f"{mode}_{name}", metric) for name, metric in metrics.items()])
    # m_vals[f"{mode}_acc_wg"] = min([metric.compute() for name, metric in metrics.items() if name.find("_g") > 0])
    # m_vals[f"{mode}_acc_avg"] = \
    #     sum([metric.compute() for name, metric in metrics.items() if name.find("_g") > 0])/len(metrics)
    return m_vals


def get_metric_vals_epoch(metrics, mode):
    m_vals = {}
    m_vals[f"{mode}_acc_wg"] = min([metric.compute() for name, metric in metrics.items() if name.find("_g") > 0])
    num_g_metrics = len([name for name, metric in metrics.items() if name.find("_g") > 0])
    m_vals[f"{mode}_acc_avg"] = \
        sum([metric.compute() for name, metric in metrics.items() if name.find("_g") > 0])/num_g_metrics
    return m_vals


class LitClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_network(self.hparams.network_name, self.hparams.network_kwargs,
                                 self.hparams.initialization_factor)
        self.class_weights = self.hparams.class_weights #kwargs["class_weights"]

        # self.model = BoundSequential.convert(self.model)
        populate_metrics(self, self.hparams.num_classes, self.hparams.num_groups)

    def forward(self, x):
        x = self.model(x)
        return x
    
    @staticmethod
    def loss(x, y, class_weights=None):
        return F.cross_entropy(x, y, weight=class_weights.to(x.device))

    def training_step(self, batch, batch_idx):
        loss = self.default_step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        return self.default_step(batch, mode="valid")

    def test_step(self, batch, batch_idx):
        return self.default_step(batch, mode="test")

    def default_step(self, batch, mode):
        x, y, m, g = batch
        y_hat = self(x)
        # y_hat = self.model(x, method_opt="forward", disable_multi_gpu=False)
        probs = torch.softmax(y_hat, dim=-1)
        loss = self.loss(y_hat, y, class_weights=self.class_weights)
        # print(y_hat, y, loss)

        with torch.no_grad():
            active_mask = torch.reshape(m, [len(m), -1]).sum(dim=-1) > 0
            if False: # active_mask.sum() > 0:
                regval, logit_diff, prob_diff = AttributionIBPRegularizer(self.model, x[active_mask], y[active_mask],
                                                                          m[active_mask], 1e-1,
                                                                          num_class=self.hparams.num_classes, mode=mode)
            else:
                regval, logit_diff, prob_diff = 0, 0, 0

        metrics = self.train_metrics if mode == "train" else self.valid_metrics
        update_metrics(metrics, probs, y, g, self.hparams.num_groups)

        m_vals = get_metric_vals(metrics, mode)
        # log calls metric reset by default
        self.log_dict({
            f"{mode}_loss": loss,
            f"{mode}_regval": regval,
            f"{mode}_logit_diff_abs_mean": logit_diff,
            f"{mode}_prob_diff_abs_mean": prob_diff,
            **m_vals
        }, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        assert self.hparams.optimizer in ["sgd", "adam", "adamw"], "Wrong optimizer name"

        if self.hparams.optimizer == "sgd":
            optim = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adam":
            optim = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamw":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
            )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optim, milestones=self.hparams.milestones, gamma=0.1
            ),
            "name": "lr_history",
        }

        return [optim], [scheduler]

    def training_epoch_end(self, outputs) -> None:
        m_vals = get_metric_vals_epoch(self.train_metrics, 'train')
        self.log_dict(m_vals)

    def validation_epoch_end(self, outputs):
        m_vals = get_metric_vals_epoch(self.valid_metrics, 'valid')
        self.log_dict(m_vals)
        print("Validation results")
        print(m_vals)

    def test_epoch_end(self, outputs):
        m_vals = get_metric_vals_epoch(self.valid_metrics, 'test')
        self.log_dict(m_vals)
