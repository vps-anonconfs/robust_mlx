import torch

from .utils.Interpreter import Interpreter
from .utils.metrics import get_accuracy
from .base_trainer import LitClassifier, update_metrics, get_metric_vals, get_metric_vals_epoch

from .utils.IBP_conv_functions import AttributionIBPRegularizer, BoundSequential


class LitRRRClassifier(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.interpreter = Interpreter(self.model)

    def valid_test_step(self, batch, mode):

        x, y, m, g = batch

        is_enable_grad = torch.enable_grad if mode=='train' else torch.no_grad

        with torch.no_grad():
            # forward
            y_hat = self(x)

            # cross entropy loss
            ce_loss = self.loss(y_hat, y, class_weights=self.class_weights)

            # evaluation
            probs = torch.softmax(y_hat, dim=-1)
            metrics = self.train_metrics if mode == "train" else self.valid_metrics

            update_metrics(metrics, probs, y, g, self.hparams.num_groups)
            m_vals = get_metric_vals(metrics, mode)

            self.log_dict(
                    {
                        f"{mode}_ce_loss": ce_loss,
                        **m_vals
                    },
                    prog_bar=True,
                    on_epoch=True,
                )
        return ce_loss


    def default_step(self, batch, mode):
        if mode in ["test", "valid"]:
            return self.valid_test_step(batch, mode)

        x, y, m, g = batch

        with torch.enable_grad ():
            x.requires_grad = True

            # forward
            y_hat = self(x)

            # cross entropy loss
            ce_loss = self.loss(y_hat, y, class_weights=self.class_weights)

            # attribution prior loss
            h = self.interpreter.get_heatmap(
                x,
                y,
                y_hat,
                method=self.hparams.rrr_hm_method,
                normalization=self.hparams.rrr_hm_norm,
                threshold=self.hparams.rrr_hm_thres,
                trainable=True,
                hparams=self.hparams,
            )
            # ap_loss = (h * m).abs().sum()
            ap_loss = torch.norm((h * m), p=2, dim=(1,2,3)).sum() # L2 norm

            # total loss
            loss = ce_loss + self.hparams.rrr_ap_lamb * ap_loss

        with torch.no_grad():
            # evaluation
            probs = torch.softmax(y_hat, dim=-1)
            metrics = self.train_metrics if mode == "train" else self.valid_metrics

            update_metrics(metrics, probs, y, g, self.hparams.num_groups)
            m_vals = get_metric_vals(metrics, mode)

            self.log_dict(
                    {
                        # f"{mode}_spuri_grad": (h_clone*m).mean(),
                        # f"{mode}_core_grad": (h_clone*(neg_m)).mean(),
                        f"{mode}_loss": loss,
                        f"{mode}_ce_loss": ce_loss,
                        f"{mode}_ap_loss": ap_loss,
                        **m_vals
                    },
                    prog_bar=True,
                    on_epoch=True,
                )
        return loss

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
     
