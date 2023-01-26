import torch
import copy
from .utils.metrics import get_auc
from src.networks.model_types import four_layer_cnn

import pytorch_lightning as pl


class IBP_Test(pl.LightningModule):
    def __init__(
        self,
        exp_id,
    ):
        super().__init__()
        self.save_hyperparameters()
       
        self.ckpt = self.load_ckpt(exp_id)
        self.model = four_layer_cnn(3, 299, 8, 512) #IBP_large(3, 299)
        self.safe_model_loader(self.model, self.ckpt)
#         self.model = BoundSequential.convert(self.model)
        self.model = self.model.cuda()
        self.model.eval()

        # This is for auc : save prediction for all batches
        self.y_hat_class1_list = []
        self.y_list = []
        self.y_hat_L_class1_list = {}
        self.y_hat_U_class1_list = {}

        self.eps_list = [0.5, 1.0]

    def safe_model_loader(self, model, ckpt):
        new_ckpt = {}
        for i in list(ckpt["state_dict"].keys()):
            if "model" in i:
                model_parameter_name = i[6:]
                new_ckpt[str(model_parameter_name)] = ckpt["state_dict"][i]
        model.load_state_dict(new_ckpt)
        return

    def load_ckpt(self, exp_id):
        path = "/mnt/ssd/jj/Research/IBP_based_AP_XAI/IBP_AP/output/IBP2/" + str(exp_id) + "/last.ckpt"
        return torch.load(path, map_location="cuda")

    def validation_step(self, batch, batch_idx):
        self.default_step(batch, batch_idx)
        return 
    
    def on_validation_epoch_end(self):
        self.get_auc()

    def get_auc(self, mode='test_on_trained_model'):
        if len(self.y_list)>5: # This is for avoiding sanity check when starting to run code
            
            y_hat = torch.cat((self.y_hat_class1_list),0)
            y = torch.cat((self.y_list),0)

            roc_auc_score_, f1 = get_auc(y, y_hat)
            self.log( f"{mode}_roc_auc_score", roc_auc_score_)
            self.log( f"{mode}_f1_CDEPcode", f1)

            for eps in self.eps_list:
                y_hat_L = torch.cat((self.y_hat_L_class1_list[eps]),0)
                y_hat_U = torch.cat((self.y_hat_U_class1_list[eps]),0)
                roc_auc_score_L, f1_L = get_auc(y, y_hat)
                roc_auc_score_U, f1_U = get_auc(y, y_hat)

                self.log( f"{mode}_L_roc_auc_score", roc_auc_score_L)
                self.log( f"{mode}_L_roc_auc_score", roc_auc_score_U)
                self.log( f"{mode}_U_f1_CDEPcode", f1_L)
                self.log( f"{mode}_U_f1_CDEPcode", f1_U)

            # Reset lists
            self.y_hat_class1_list=[]
            self.y_list=[]
            self.y_hat_L_class1_list={}
            self.y_hat_U_class1_list={}
            return 

    def default_step(self, batch, batch_idx):
        x, y, m = batch
        
        # Model foward for normal model (not IBP model)
        y_hat = self.model(x) #, method_opt="forward",disable_multi_gpu=False)
        y_hat_s = soft(self.model(x_L))[:, 1].detach().cpu()
        self.y_hat_class1_list.append(y_hat_s)
        self.y_list.append(y.detach().cpu())

        # Make perturbation on sticker parts
        soft = torch.nn.Softmax(dim=1)
        eps_list = []
        for eps in self.eps_list:
            self.y_hat_L_class1_list[eps]=[]
            self.y_hat_U_class1_list[eps]=[]
        for eps in self.eps_list:
            x_L = x - (m * eps)
            x_U = x + (m * eps)
            y_hat_L = soft(self.model(x_L))[:, 1].detach().cpu()
            y_hat_U = soft(self.model(x_U))[:, 1].detach().cpu()
            pred_diff = (y_hat_U - y_hat_U).abs()

            self.y_hat_L_class1_list[eps].append(y_hat_L)
            self.y_hat_U_class1_list[eps].append(y_hat_U)

            self.log( f"{str(eps)}_eps_pred_diff", pred_diff, on_step=False, on_epoch=True)

        # # Set interval: we use mask 
        # self.fair_interval = m
        
        
        # regval = torch.tensor(0.0)
        # eps = self.eps if mode=='train' else self.hparams.EPSILON
        # if self.hparams.ALPHA > 0:
        #     regval = self.AttributionIBPRegularizer(self.model, x, y, self.fair_interval, self.hparams.EPSILON)
            
        # loss = ((1-self.hparams.ALPHA) * ce_loss) + (self.hparams.ALPHA * regval)
        # self.log_dict(
        #         {
        #             f"{mode}_auc": auc,
        #         },
        #         prog_bar=True,
        #     )
        # return auc


    def step_for_jupyter(self, batch, batch_idx):
        x, y, m = batch
        
        # Model foward for IBP
        y_hat = self.model(x) #, method_opt="forward",disable_multi_gpu=False)
        return y_hat.argmax(1)
    
    def pgd_for_batch(self, batch, batch_idx, iters=25):
        x, y, m = batch
        adv = copy.deepcopy(x)
        loss = torch.nn.CrossEntropyLoss()
        for i in range(iters):
            adv.requires_grad = True
            y_hat = self.model(adv)
            model.zero_grad()
            cost = loss(outputs, lab)
            cost.backward()
            adv = adv_inp + (eps/iters)*(adv.grad.sign()*m)
            adv = torch.clamp(adv, min=0, max=1).detach_()
        return adv
