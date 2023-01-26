# Imports
import copy
import torch
import numpy as np
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.utils.multiclass import type_of_target
from sklearn.decomposition import TruncatedSVD
from torch.nn import functional as F

GLOBAL_MODEL_PETURB = 'relative'
SOMEWHAT_BIG_NUMBER = 10

"""
=============================================================
Interval Propagation Code for Learning Certifiably Fair NNs
=============================================================
"""


# Define forward propagation through a nn
def affine_forward(W, b, x_l, x_u, marg=0, b_marg=0):
    """
    This function uses pytorch to compute upper and lower bounds
    on a matrix multiplication given bounds on the matrix 'x' 
    as given by x_l (lower) and x_u (upper)
    """
    marg = marg / 2;
    b_marg = b_marg / 2
    x_mu = (x_u + x_l) / 2
    x_r = (x_u - x_l) / 2
    W_mu = W
    if (GLOBAL_MODEL_PETURB == 'relative'):
        W_r = torch.abs(W) * marg
    elif (GLOBAL_MODEL_PETURB == 'absolute'):
        W_r = torch.ones_like(W) * marg
    b_u = torch.add(b, b_marg)
    b_l = torch.subtract(b, b_marg)
    h_mu = torch.matmul(x_mu, W_mu.T)
    x_rad = torch.matmul(x_r, torch.abs(W_mu).T)
    # assert((x_rad >= 0).all())
    W_rad = torch.matmul(torch.abs(x_mu), W_r.T)
    # assert((W_rad >= 0).all())
    Quad = torch.matmul(torch.abs(x_r), torch.abs(W_r).T)
    # assert((Quad >= 0).all())
    h_u = torch.add(torch.add(torch.add(torch.add(h_mu, x_rad), W_rad), Quad), b_u)
    h_l = torch.add(torch.subtract(torch.subtract(torch.subtract(h_mu, x_rad), W_rad), Quad), b_l)
    return h_l, h_u


def interval_bound_forward(model, weights, inp, vec, eps):
    h_l = inp - (vec * eps);
    h_u = inp + (vec * eps)
    assert ((h_l <= h_u).all())
    # h_l = torch.clip(h_l, 0, 1);
    # h_u = torch.clip(h_u, 0, 1)
    num_layers = len(model.layers)  # int(len(weights) / 2);
    for i in range(len(model.layers)):
        if "LINEAR" in model.layers[i].upper():
            w, b = weights[2 * (i)], weights[(2 * (i)) + 1]
            h_l, h_u = affine_forward(w, b, h_l, h_u, marg=0.0, b_marg=0.0)
            # assert((h_l <= h_u).all())
            if i < num_layers - 1:  # Return Logits not Softmax Activation
                h_l = model.activations[i](h_l)
                h_u = model.activations[i](h_u)
        else:
            # Can pull convolutional layers over from other Cert Modules
            assert False, "Layers that are note linear layers are not supported at this time"
    return h_l, h_u


def AttributionIBPRegularizer(model, inp, lab, vec, eps, nclasses):
    inp_dim = inp.shape[-1]
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    v1 = lab
    v2 = 1 - v1
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    return F.cross_entropy(worst_case, lab)


# Define standard robustness bounds as a sanity check

def fairness_regularizer(model, inp, lab, vec, eps, nclasses=10):
    """
    This class only works for binary classification at the moment. Can be generalized
    with a bit of effort modifying the for loop.
    """
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    worst_delta = 0
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    min_i_softmax = F.softmax(min_logit, dim=-1)
    max_i_softmax = F.softmax(max_logit, dim=-1)
    worst_delta = torch.sum(torch.abs(max_i_softmax - min_i_softmax))
    return worst_delta  # F.cross_entropy(worst_case, lab)


def fairness_bounds(model, inp, lab, vec, eps, nclasses):
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    best_case = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    y_l = F.softmax(worst_case, dim=-1)
    y_u = F.softmax(best_case, dim=-2)
    return y_l, y_u


def fairness_delta(model, inp, lab, vec, eps, nclasses):
    """
    This class only works for binary classification at the moment. Can be generalized
    with a bit of effort modifying the for loop.
    """
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    worst_delta = 0
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    min_i_softmax = F.softmax(min_logit, dim=-1)
    max_i_softmax = F.softmax(max_logit, dim=-1)
    delta = (max_i_softmax - min_i_softmax)
    delta = delta.detach().numpy()
    return np.max(delta, axis=1)


def fair_PGD(model, x_natural, lab, vec, eps, nclasses, iterations=10):
    x = x_natural.detach()
    eps_vec = vec*eps
    noise = (-2*vec) * torch.zeros_like(x).uniform_(0, 1) + vec
    x = x + (noise*eps)
    for i in range(iterations):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, lab)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + 0.5 * torch.sign(grad.detach())
        x = torch.min(torch.max(x, x_natural - eps_vec), x_natural + eps_vec)
    return x 

def fairness_delta_PGD(model, inp, lab, vec, eps, nclasses, iterations=10):
    y_pred = model(inp)
    x_adv = fair_PGD(model, inp, lab, vec, eps, nclasses, iterations)
    y_adv = model(x_adv)
    pgd_delta = torch.max(torch.abs(y_pred - y_adv), axis=1)
    return pgd_delta

def fairness_regularizer_PGD(model, inp, lab, vec, eps, nclasses, iterations=10):
    y_pred = model(inp)
    x_adv = fair_PGD(model, inp, lab, vec, eps, nclasses, iterations)
    y_adv = model(x_adv)
    regval = torch.sum(torch.abs(y_pred - y_adv))
    return regval
    
def RobustnessRegularizer(model, inp, lab, vec, eps, nclasses):
    inp_dim = inp.shape[-1]
    # vec = torch.ones([inp_dim]).to(inp.device).double()
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - v1
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    return F.cross_entropy(worst_case, lab)


def RobustnessBounds(model, inp, lab, vec, eps, nclasses):
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    best_case = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    y_l = worst_case
    y_u = best_case
    return y_l, y_u