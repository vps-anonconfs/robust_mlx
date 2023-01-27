import torch
import torchmetrics
import copy

from .utils.IBP_conv_functions import AttributionIBPRegularizer, BoundSequential
from .base_trainer import LitClassifier, update_metrics, get_metric_vals, get_metric_vals_epoch
from src.networks import get_network



class LitCoRMClassifier(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.hparams.max_epochs > 0, f"max_epochs {self.hparams.max_epochs} needs to be positive"
        self.model = get_network(self.hparams.network_name, self.hparams.network_kwargs,
                                 self.hparams.initialization_factor)
        self.eps = self.hparams.corm_EPSILON


    def default_step(self, batch, mode):
        x, y, m, g = batch

        # Model foward for normal model (not IBP model)
        y_hat = self.model(x)
        ce_loss = self.loss(y_hat, y, class_weights=self.class_weights)
        probs = torch.softmax(y_hat, dim=-1)
        metrics = self.train_metrics if mode == "train" else self.valid_metrics
        update_metrics(metrics, probs, y, g, self.hparams.num_groups)
        loss = ce_loss

        noise_ce_loss = 0
        if mode in ["train"]:
            # Gaussian noise
            noise = torch.normal(0, self.eps, size=x.shape).cuda()
            x_n = x + (m)*noise
            y_hat_n = self.model(x_n)
            noise_ce_loss = self.loss(y_hat_n, y, class_weights=self.class_weights)
            loss += noise_ce_loss

        m_vals = get_metric_vals(metrics, mode)
        self.log_dict(
                {
                    f"{mode}_loss": loss,
                    f"{mode}_ce_loss": ce_loss,
                    f"{mode}_noise_ce_loss": noise_ce_loss,
                    **m_vals, 
                },
                prog_bar=True,
                on_epoch=True,
            )
        return loss