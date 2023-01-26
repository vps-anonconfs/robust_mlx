import torch

from .utils.IBP_conv_functions import AttributionIBPRegularizer, BoundSequential
from .base_trainer import LitClassifier, update_metrics, get_metric_vals, get_metric_vals_epoch
from src.networks import get_network
from .utils.Interpreter import Interpreter


class LitIBPClassifier(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.eps = self.hparams.ibp_start_EPSILON
        self.alpha = self.hparams.ibp_ALPHA

        assert self.hparams.max_epochs > 0, f"max_epochs {self.hparams.max_epochs} needs to be positive"
        self.model = get_network(self.hparams.network_name, self.hparams.network_kwargs,
                                 self.hparams.initialization_factor)
        self.model = BoundSequential.convert(self.model)
        self.interpreter = Interpreter(self.model)

    def default_step(self, batch, mode):
        x, y, m, g = batch
        # Model forward for IBP
        y_hat = self.model(x, method_opt="forward", disable_multi_gpu=False)
        ce_loss = self.loss(y_hat, y, class_weights=self.class_weights)

        # eps = self.eps if mode == 'train' else self.hparams.ibp_EPSILON
        alpha = self.alpha if mode == 'train' else self.hparams.ibp_ALPHA

        # for efficiency concerns
        active_mask = torch.reshape(m, [len(m), -1]).abs().sum(dim=-1) > 0
        if active_mask.sum() > 0:
            regval, logit_diff, prob_diff = AttributionIBPRegularizer(self.model, x[active_mask], y[active_mask],
                                                                      m[active_mask], self.eps,
                                                                      num_class=self.hparams.num_classes, mode=mode)
            with torch.no_grad():
                _, logit_diff_debug, prob_diff_debug = AttributionIBPRegularizer(self.model, x[active_mask],
                                                                                 y[active_mask], m[active_mask], 0.1,
                                                                                 num_class=self.hparams.num_classes, mode=mode)
        else:
            regval, logit_diff, prob_diff = 0, 0, 0
            logit_diff_debug, prob_diff_debug = 0, 0

        loss = (alpha * ce_loss) + (1 - alpha) * regval

        if self.hparams.ibp_rrr > 0:
            with torch.enable_grad():
                x.requires_grad = True
                _y_hat = self.model(x, method_opt="forward", disable_multi_gpu=False)
                # attribution prior loss
                h = self.interpreter.get_heatmap(
                    x,
                    y,
                    _y_hat,
                    method=self.hparams.rrr_hm_method,
                    normalization=self.hparams.rrr_hm_norm,
                    threshold=self.hparams.rrr_hm_thres,
                    trainable=True,
                    hparams=self.hparams,
                )
                loss += self.hparams.ibp_rrr*(h*m).abs().sum()

        probs = torch.softmax(y_hat, dim=-1)
        metrics = self.train_metrics if mode == "train" else self.valid_metrics
        update_metrics(metrics, probs, y, g, self.hparams.num_groups)

        m_vals = get_metric_vals(metrics, mode)
        self.log_dict(
                {
                    f"{mode}_prob_diff_abs_mean": prob_diff_debug,
                    f"{mode}_logit_diff_abs_mean": logit_diff_debug,
                    f"{mode}_loss": loss,
                    f"{mode}_ce_loss": ce_loss,
                    f"{mode}_regval": regval,
                    **m_vals
                },
                prog_bar=True,
                on_epoch=True,
            )
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.alpha -= 0.5*self.hparams.ibp_ALPHA/self.hparams.max_epochs
        self.eps += self.hparams.ibp_EPSILON/self.hparams.max_epochs
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