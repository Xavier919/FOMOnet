#imports
import torch
import torch.optim as optim
from torch import nn
import pickle
import numpy as np
#project specific imports
import argparse
from matplotlib import pyplot as plt
from torchmetrics import PrecisionRecallCurve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from torch import cat
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('reports')
parser.add_argument('bins')
parser.add_argument('tag', type=str)
args = parser.parse_args()

if __name__ == "__main__":

    reports = pickle.load(open(args.reports, 'rb'))

    #preds_ = [x['out'] for x in reports.values()]
    #target_ = [x['mapped_cds'] for x in reports.values()]

    #preds = cat(preds_)
    #target = cat(target_).long()
    #pr_curve = PrecisionRecallCurve(pos_label=1,task="binary")
    #precision, recall, _ = pr_curve(preds, target)
    #auc_pr = auc(recall, precision)
    #plt.plot(recall, precision, color = 'black')
    #plt.ylim(0.9, 1.05), plt.xlim(0, 1)
    #plt.xlabel("Recall"), plt.ylabel("Precision"), plt.title('PR curve')
    #plt.legend(['PR AUC: {}'.format(round(auc_pr, 3))])
    #plt.savefig(f'pr_curve{args.tag}.svg')
    #plt.clf()

    #preds = cat(preds_).detach().numpy()
    #target = cat(target_).long().detach().numpy()
    #fpr, tpr, _ = metrics.roc_curve(target, preds)
    #roc_auc = auc(fpr, tpr)
    #plt.plot(fpr, tpr, color = 'black')
    #plt.ylim(0.9, 1.05), plt.xlim(0, 1)
    #plt.xlabel("FPR"), plt.ylabel("TPR"), plt.title('ROC curve')
    #plt.legend(['ROC AUC: {}'.format(round(roc_auc, 3))])
    #plt.savefig(f'roc_curve{args.tag}.svg')
    #plt.clf()




    for bin_ in args.bins:
        preds = [y['out'] for x,y in reports.items() if x in bin_]
        target = [y['mapped_cds'] for x,y in reports.items() if x in bin_]
        preds = cat(preds).detach().numpy()
        target = cat(target).long().detach().numpy()
        fpr, tpr, _ = metrics.roc_curve(target, preds)
        roc_auc = auc(fpr, tpr)
        plt.scatter(fpr, tpr, color = 'grey')
        
    preds = [y['out'] for x,y in reports.items()]
    target = [y['mapped_cds'] for x,y in reports.items()]
    preds = cat(preds).detach().numpy()
    target = cat(target).long().detach().numpy()
    fpr, tpr, _ = metrics.roc_curve(target, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color = 'green')
    
    plt.ylim(0.9, 1.05), plt.xlim(0, 1)
    plt.xlabel("FPR"), plt.ylabel("TPR"), plt.title('ROC curve')
    plt.legend(['ROC AUC: {}'.format(round(roc_auc, 3))])
    plt.savefig(f'roc_curve{args.tag}.svg')
    plt.clf()



    for bin_ in args.bins:
        preds = [y['out'] for x,y in reports.items() if x in bin_]
        target = [y['mapped_cds'] for x,y in reports.items() if x in bin_]
        preds = cat(preds_)
        target = cat(target_).long()
        pr_curve = PrecisionRecallCurve(pos_label=1,task="binary")
        auc_pr = auc(recall, precision)
        plt.scatter(fpr, tpr, color = 'grey')
        
    preds = cat(preds_)
    target = cat(target_).long()
    pr_curve = PrecisionRecallCurve(pos_label=1,task="binary")
    precision, recall, _ = pr_curve(preds, target)
    auc_pr = auc(recall, precision)
    plt.plot(recall, precision, color = 'green')
    
    plt.ylim(0.9, 1.05), plt.xlim(0, 1)
    plt.xlabel("Recall"), plt.ylabel("Precision"), plt.title('PR curve')
    plt.legend(['PR AUC: {}'.format(round(auc_pr, 3))])
    plt.savefig(f'pr_curve{args.tag}.svg')
    plt.clf()