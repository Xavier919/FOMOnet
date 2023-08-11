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
#project specific imports
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('preds1')
parser.add_argument('preds2')
parser.add_argument('preds3')
parser.add_argument('preds4')
parser.add_argument('preds5')
parser.add_argument('preds6')
args = parser.parse_args()


def pr_curve(preds, target):
    preds = cat(preds)
    target = cat(target).long()
    pr_curve = PrecisionRecallCurve(pos_label=1,task="binary")
    precision, recall, _ = pr_curve(preds, target)
    auc_pr = auc(recall, precision)
    plt.plot(recall, precision, color = 'black')
    plt.ylim(0, 1), plt.xlim(0, 1)
    plt.xlabel("Recall"), plt.ylabel("Precision"), plt.title('PR curve')
    plt.legend(['PR AUC: {}'.format(round(auc_pr, 3))])
    plt.savefig('pr_curve.svg')
    plt.show()
    
def roc_curve(preds, target):
    preds = cat(preds).detach().numpy()
    target = cat(target).long().detach().numpy()
    fpr, tpr, _ = metrics.roc_curve(target, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color = 'black')
    plt.ylim(0, 1), plt.xlim(0, 1)
    plt.xlabel("FPR"), plt.ylabel("TPR"), plt.title('ROC curve')
    plt.legend(['ROC AUC: {}'.format(round(roc_auc, 3))])
    plt.savefig('roc_curve.svg')
    plt.show()

if __name__ == "__main__":

    preds1 = pickle.load(open(args.preds1, 'rb'))
    preds2 = pickle.load(open(args.preds2, 'rb'))
    preds3 = pickle.load(open(args.preds3, 'rb'))
    preds4 = pickle.load(open(args.preds4, 'rb'))
    preds5 = pickle.load(open(args.preds5, 'rb'))
    preds6 = pickle.load(open(args.preds6, 'rb'))



