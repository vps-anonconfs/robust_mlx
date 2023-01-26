import torch


from .base_trainer import LitClassifier, update_metrics, get_metric_vals
from src.networks import get_network
from .utils import CDEP_utils


class LitCDEPClassifier(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cdep_ap_lamb = kwargs['cdep_ap_lamb']
        
        # load model
        self.model = get_network(self.hparams.network_name, self.hparams.network_kwargs,
                                 self.hparams.initialization_factor)

    def default_step(self, batch, mode):
        x, y, mask, g = batch

        mask = mask.type(x.dtype)
        # Model forward for CDEP
        y_hat = self.model(x)
        ce_loss = self.loss(y_hat, y, class_weights=self.class_weights)

        # regularizer
        cdep_loss = 0
        if mask.any():
            rel, irrel = CDEP_utils.cd(1-mask, x, self.model, device=x.device)
            # just suppressing irrelevant logits
            cur_cd_loss = torch.nn.functional.softmax(torch.stack((rel[:, 0], irrel[:, 0]), dim=1), dim=1)[:, 0].mean()
            cur_cd_loss += torch.nn.functional.softmax(torch.stack((rel[:, 1], irrel[:, 1]), dim=1), dim=1)[:, 0].mean()
            cdep_loss = cur_cd_loss / 2

        loss = ce_loss + self.cdep_ap_lamb * cdep_loss

        # Evaluation
        probs = torch.softmax(y_hat, dim=-1)
        metrics = self.train_metrics if mode == "train" else self.valid_metrics
        update_metrics(metrics, probs, y, g, self.hparams.num_groups)

        m_vals = get_metric_vals(metrics, mode)
        self.log_dict(
                {
                    f"{mode}_loss": loss,
                    f"{mode}_ce_loss": ce_loss,
                    f"{mode}_cdep_loss": cdep_loss,
                    **m_vals
                },
                prog_bar=True,
                on_epoch=True,
            )
        return loss
