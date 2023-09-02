#imports
import torch
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from torch import cat
from sklearn import metrics
from torchmetrics import PrecisionRecallCurve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('split1')
parser.add_argument('split2')
parser.add_argument('split3')
parser.add_argument('split4')
parser.add_argument('split5')
parser.add_argument('split6')

args = parser.parse_args()

def PR_curve(list_preds, list_targets):
    pr_curve = PrecisionRecallCurve(pos_label=1)
    #mean curve
    cat_preds = cat([x.flatten() for y in list_preds for x in y])
    cat_targets = cat([x.flatten() for y in list_targets for x in y]).long()
    precision, recall, _ = pr_curve(cat_preds, cat_targets)
    auc_pr = auc(recall, precision)
    plt.plot(recall, precision, color = 'black', linewidth=1)
    #individual curves
    for preds, targets in zip(list_preds, list_targets):
        cat_preds = cat([x.flatten() for x in preds])
        cat_targets = cat([x.flatten() for x in targets]).long()
        precision, recall, _ = pr_curve(cat_preds, cat_targets)
        plt.scatter(recall, precision, color = 'green', s=0.05)
    
    plt.ylim(0.95, 1.01), plt.xlim(0, 1)
    plt.xlabel("recall"), plt.ylabel("precision"), plt.title('precision-recall curve')
    plt.legend(['PR auc: {}'.format(round(auc_pr, 10))])
    plt.savefig('pr_curve.svg')
    plt.show()
    plt.clf()

    def ROC_curve(list_preds, list_targets):
        #mean curve
        cat_preds = cat([x.flatten() for y in list_preds for x in y]).detach().numpy()
        cat_targets = cat([x.flatten() for y in list_targets for x in y]).long().detach().numpy()
        fpr, tpr, _ = metrics.roc_curve(cat_targets, cat_preds)
        auc_roc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color = 'black', linewidth=1)
        #individual curves
        for preds, targets in zip(list_preds, list_targets):
            cat_preds = cat([x.flatten() for x in preds]).detach().numpy()
            cat_targets = cat([x.flatten() for x in targets]).long().detach().numpy()
            fpr, tpr, _ = metrics.roc_curve(cat_targets, cat_preds)
            plt.scatter(fpr, tpr, color = 'green', s=0.05)
        
        plt.ylim(0.95, 1.01), plt.xlim(0, 1)
        plt.xlabel("FPR"), plt.ylabel("TPR"), plt.title('ROC curve')
        plt.legend(['ROC auc: {}'.format(round(auc_roc, 10))])
        plt.savefig('roc_curve.svg')
        plt.show()
        plt.clf()

if __name__ == "__main__":
    split1 = pickle.load(open(args.split1, 'rb'))
    split2 = pickle.load(open(args.split2, 'rb'))
    split3 = pickle.load(open(args.split3, 'rb'))
    split4 = pickle.load(open(args.split4, 'rb'))
    split5 = pickle.load(open(args.split5, 'rb'))
    split6 = pickle.load(open(args.split6, 'rb'))

    preds = [split1[0], split2[0], split3[0], split4[0], split5[0], split6[0]]
    targets = [split1[1], split2[1], split3[1], split4[1], split5[1], split6[1]]

    PR_curve(preds, targets)
    ROC_curve(preds, targets)