import torch
class Interpreter:
    def __init__(self, model):
        self.model = model

    def get_heatmap(self, x, y, y_hat, method, normalization="absmax", threshold="abs", trainable=False, hparams=None):
        
        if x.requires_grad == False:
            x.requires_grad = True
        
        # Autograd
        #y_hat_c = y_hat[range(len(y_)), y_] # For classification
        if method == 'rrr':
            y_hat_c = y_hat  # For regression
            return self.simple_grad(x, y_hat_c, trainable=trainable)
        else:
            y_hat_c = y_hat  # For regression
            h = self.simple_grad(x, y_hat_c, trainable=trainable)

        # reduction
        if len(h.shape) == 4:
            h = h.sum(dim=1)
        
        # threshold
        if threshold == "abs":
            h = h.abs()

        # normalization
        if normalization == "standard":
            h_max = h.max(dim=2, keepdims=True)[0].max(dim=1, keepdims=True)[0]
            h = h / (h_max + 1e-8)
        elif normalization == "absmax":
            h_max = h.abs().max(dim=2, keepdims=True)[0].max(dim=1, keepdims=True)[0]
            h = h / (h_max + 1e-8)
        elif normalization == "sum":
            h = h / (h.sum(dim=(1, 2), keepdims=True) + 1e-8)
            
        return h
    
    def simple_grad_old(self, x, y_hat_c, trainable=False):
        h = torch.autograd.grad(
            y_hat_c, x, grad_outputs=torch.ones_like(y_hat_c), create_graph=trainable, retain_graph=trainable,
        )[0]
        return h
    
    def simple_grad(self, x, y_hat_c, trainable=False):
        if (len(y_hat_c.shape) > 1) and (y_hat_c.shape[1] == 2):
            h = [torch.autograd.grad(
                y_hat_c[:, i], x, grad_outputs=torch.ones_like(y_hat_c[:, i]), create_graph=trainable, retain_graph=trainable,
            )[0] for i in range(y_hat_c.shape[1])]
            h = torch.stack([_h**2 for _h in h], dim=-1).sum(dim=-1)
        else:
            h = torch.autograd.grad(
                y_hat_c, x, grad_outputs=torch.ones_like(y_hat_c), create_graph=trainable, retain_graph=trainable,
            )[0]
        return h