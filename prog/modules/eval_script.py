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
#project specific imports
from model import FOMOnet
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data')
parser.add_argument('model')
parser.add_argument('kernel', type=int)
parser.add_argument('trxps', type=str)
parser.add_argument('tag', type=str)
args = parser.parse_args()


def bin_pred(output, thresh):
    bin_pred = (output>thresh).int()
    return bin_pred

def iou_score(target, output):
    target = target.flatten()
    output = output.flatten()
    intersection = (target * output).sum()
    union = target.sum() + output.sum() - intersection
    iou = intersection / union
    return iou.item()

def recall_score(target, output):
    target = target.flatten()
    output = output.flatten()
    true_positives = ((target == 1) & (output == 1)).sum()
    false_negatives = ((target == 1) & (output == 0)).sum()
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    return recall.item()

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

def get_preds(model, X_test):
    preds = []
    model.eval()
    for X in X_test:
        pad = torch.zeros(4,5000)
        X_ = torch.cat([pad,X,pad],dim=1).view(1,4,-1)
        out = model(X_).view(-1)
        preds.append(out[pad.shape[1]:-pad.shape[1]].cpu().detach())
    return preds

def get_report(preds, y_test, trxps):
    report = dict()
    for idx, out in enumerate(preds):
        trx = trxps[idx]
        target = y_test[idx].view(-1)
        pred = bin_pred(out, 0.5)
        recall = recall_score(target, pred)
        iou = iou_score(target, pred)
        report[trx] = {'out': out,
                       'mapped_cds': target,
                       'iou': iou,
                       'recall': recall}
    return report

if __name__ == "__main__":

    X_test, y_test = pickle.load(open(args.data, 'rb'))

    X_test = [map_seq(x) for x in X_test]

    trxps = pickle.load(open(args.trxps, 'rb'))
    _, trxps, _, _ = train_test_split(trxps, trxps, test_size=0.1, random_state=42)

    #fomonet = FOMOnet(k=args.kernel)
    #fomonet.load_state_dict(torch.load(args.model, map_location=torch.device('cuda')))

    fomonet = FOMOnet(k=args.kernel)
    state_dict = torch.load(args.model, map_location=torch.device('cuda'))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # Remove "module." prefix
    fomonet.load_state_dict(state_dict)

    preds = get_preds(fomonet, X_test)

    report = get_report(preds, y_test, trxps)

    pickle.dump(preds, open(f'preds_{args.tag}.pkl', 'wb'))
    pickle.dump(report, open(f'report_{args.tag}.pkl', 'wb'))