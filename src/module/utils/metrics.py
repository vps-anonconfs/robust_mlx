import numpy as np
from sklearn import metrics
from sklearn.metrics import auc, average_precision_score, roc_curve, roc_auc_score, precision_recall_curve, f1_score

def get_accuracy(y_hat, y):
        return (y_hat.argmax(dim=1) == y).sum().item() * 1.0 / len(y)
        # return (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item() * 1.0 / len(y)
#         return (y_hat-y).sum().item()*1.0/len(y)

def get_auc(y_hat, y):
#       fpr, tpr, thresholds = metrics.roc_curve(y, y_hat, pos_label=2)
#       auc = metrics.auc(fpr, tpr)
#       print(y_hat[:5], y[:5])
      roc_auc_score_ = roc_auc_score(y_hat, y)
      f1 = np.asarray([f1_score(y_hat, y > x) for x in np.linspace(0.1, 1, num=10) if (y > x).any() and (y < x).any()]).max()
      
      return roc_auc_score_, f1

# This code is from CDEP
def get_auc_f1(y_hat, y, fname=None,):
        auc = roc_auc_score(y_hat, y)
        f1 = np.asarray([f1_score(y_hat, y > x) for x in np.linspace(0.1, 1, num=10) if (y > x).any() and (y < x).any()]).max()
        return auc, f1