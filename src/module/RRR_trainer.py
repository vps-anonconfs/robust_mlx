import torch

from .utils.Interpreter import Interpreter
from .base_trainer import LitClassifier, update_metrics, get_metric_vals, get_metric_vals_epoch

from .utils.IBP_conv_functions import AttributionIBPRegularizer, BoundSequential


class LitRRRClassifier(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.interpreter = Interpreter(self.model)

        ########## for ibp eval ############
        self.eps = self.hparams.ibp_EPSILON  # 0
        self.alpha = self.hparams.ibp_ALPHA
        ####################################

    def valid_test_step(self, batch, mode):
        ########## for ibp eval ############
        # self.eval_IBP_logit_diff(batch, mode)
        ####################################

        x, y, m, g = batch

        is_enable_grad = torch.enable_grad if mode == 'train' else torch.no_grad

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

            # for metric in metrics.values():
            #     metric.update(probs, y)
            # m_vals = dict([(f"{mode}_{name}", metric) for name, metric in metrics.items()])

            self.log_dict(
                {
                    f"{mode}_ce_loss": ce_loss,
                    **m_vals
                },
                prog_bar=True,
                on_epoch=True,
            )
        return ce_loss

    def eval_IBP_logit_diff(self, batch, mode):
        model = BoundSequential.convert(self.model).cuda()

        x, y, m, g = batch
        # Model forward for IBP
        y_hat = model(x, method_opt="forward", disable_multi_gpu=False)
        ce_loss = self.loss(y_hat, y, class_weights=self.class_weights)

        # eps = self.eps if mode == 'train' else self.hparams.ibp_EPSILON
        alpha = self.alpha if mode == 'train' else self.hparams.ibp_ALPHA

        # for efficiency concerns
        active_mask = torch.reshape(m, [len(m), -1]).abs().sum(dim=-1) > 0
        if active_mask.sum() > 0:
            with torch.no_grad():
                regval, logit_diff_debug, prob_diff_debug = AttributionIBPRegularizer(model, x[active_mask],
                                                                                 y[active_mask], m[active_mask], .1,
                                                                                 num_class=self.hparams.num_classes,
                                                                                 mode=mode)
        else:
            regval, logit_diff_debug, prob_diff_debug = 0, 0, 0

        self.log_dict(
            {
                f"{mode}_prob_diff_abs_mean": prob_diff_debug,
                f"{mode}_logit_diff_abs_mean": logit_diff_debug,
                f"{mode}_regval": regval,
            },
            prog_bar=False,
            on_epoch=True,
        )
        return

    def default_step(self, batch, mode):
        if mode in ["test", "valid"]:
            return self.valid_test_step(batch, mode)

        ########## for ibp eval ############
        # self.eval_IBP_logit_diff(batch, mode)
        ####################################
        x, y, m, g = batch

        with torch.enable_grad():
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
            ap_loss = (h * m).abs().sum()

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


