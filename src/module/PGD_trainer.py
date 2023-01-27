import torch
import torchmetrics
import copy

from .utils.IBP_conv_functions import AttributionIBPRegularizer, BoundSequential
from .base_trainer import LitClassifier, update_metrics, get_metric_vals, get_metric_vals_epoch
from src.networks import get_network

def pgd_populate_metrics(cls, num_classes, num_groups):
    task = 'multiclass'
    cls.pgd_train_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, average='macro')
    cls.pgd_valid_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, average='macro')

    cls.pgd_train_acc_per_group = torch.nn.ModuleList([torchmetrics.Accuracy(task=task, num_classes=num_classes, average='micro')
                                                   for _ in range(num_groups)])
    cls.pgd_valid_acc_per_group = torch.nn.ModuleList([torchmetrics.Accuracy(task=task, num_classes=num_classes, average='micro')
                                                   for _ in range(num_groups)])

    # PGD
    cls.pgd_train_metrics = {f'acc_g{gi}': cls.pgd_train_acc_per_group[gi] for gi in range(num_groups)}
    cls.pgd_valid_metrics = {f'acc_g{gi}': cls.pgd_valid_acc_per_group[gi] for gi in range(num_groups)}
    cls.pgd_train_metrics['acc'], cls.pgd_valid_metrics['acc'] = cls.pgd_train_acc, cls.pgd_valid_acc



class LitPGDClassifier(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.hparams.max_epochs > 0, f"max_epochs {self.hparams.max_epochs} needs to be positive"
        self.model = get_network(self.hparams.network_name, self.hparams.network_kwargs,
                                 self.hparams.initialization_factor)
        pgd_populate_metrics(self, self.hparams.num_classes, self.hparams.num_groups)

    def default_step(self, batch, mode):

        x, y, m, g = batch
 
        # Model foward for normal model (not IBP model)
        y_hat = self.model(x) #, method_opt="forward",disable_multi_gpu=False)
        ce_loss = self.loss(y_hat, y, class_weights=self.class_weights)

        # PGD
        with torch.enable_grad():
            adv = copy.deepcopy(x)
            loss = torch.nn.CrossEntropyLoss()
            eps = self.hparams.pgd_eps
            for i in range(self.hparams.pgd_iters):
                adv.requires_grad = True
                self.model.zero_grad()
                _y_hat = self.model(adv)
                cost = self.loss(_y_hat, y, class_weights=self.class_weights)
                cost.backward()
                adv = adv + ((eps/self.hparams.pgd_iters)*(adv.grad.sign()))
                adv = torch.clamp(adv, min=x-(m*eps), max=x+(m*eps)).detach_()
            pgd_y_hat = self.model(adv)
            pgd_ce_loss = self.loss(pgd_y_hat, y, class_weights=self.class_weights)
            self.model.zero_grad()

        loss = ce_loss+pgd_ce_loss

        probs = torch.softmax(y_hat, dim=-1)
        metrics = self.train_metrics if mode == "train" else self.valid_metrics
        update_metrics(metrics, probs, y, g, self.hparams.num_groups)

        # Update for pgd
        pgd_probs = torch.softmax(pgd_y_hat, dim=-1)
        pgd_metrics = self.pgd_train_metrics if mode == "train" else self.pgd_valid_metrics
        update_metrics(pgd_metrics, pgd_probs, y, g, self.hparams.num_groups)

        m_vals = get_metric_vals(metrics, mode)
        pgd_m_vals = get_metric_vals(metrics, 'pgd_'+mode)
        self.log_dict(
                {
                    f"{mode}_pgd_ce_loss": pgd_ce_loss,
                    f"{mode}_loss": loss,
                    f"{mode}_ce_loss": ce_loss,
                    **m_vals, 
                    **pgd_m_vals
                },
                prog_bar=True,
                on_epoch=True,
            )
        return loss

    def training_epoch_end(self, outputs) -> None:
        m_vals = get_metric_vals_epoch(self.train_metrics, 'train')
        pgd_m_vals = get_metric_vals_epoch(self.train_metrics, 'pgd_train')
        self.log_dict(m_vals)
        self.log_dict(pgd_m_vals)

    def validation_epoch_end(self, outputs):
        m_vals = get_metric_vals_epoch(self.valid_metrics, 'valid')
        pgd_m_vals = get_metric_vals_epoch(self.train_metrics, 'pgd_valid')
        self.log_dict(m_vals)
        self.log_dict(pgd_m_vals)
        print("Validation results")
        print(m_vals)

    def test_epoch_end(self, outputs):
        m_vals = get_metric_vals_epoch(self.valid_metrics, 'test')
        pgd_m_vals = get_metric_vals_epoch(self.train_metrics, 'pgd_test')
        self.log_dict(m_vals)
        self.log_dict(pgd_m_vals)