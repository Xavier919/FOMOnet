import numpy as np
from matplotlib import pyplot as plt
from torch import cat
from sklearn import metrics
from torchmetrics import PrecisionRecallCurve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm.notebook import tqdm
from modules.utils import *

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

def accuracy_score(target, output):
    target = target.flatten()
    output = output.flatten()
    true_positives = ((target == 1) & (output == 1)).sum()
    true_negatives = ((target == 0) & (output == 0)).sum()
    false_positives = ((target == 0) & (output == 1)).sum()
    false_negatives = ((target == 1) & (output == 0)).sum()
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives + 1e-7)
    return accuracy.item()

def find_padding_limits(tensor):
    first_non_zero = None
    last_non_zero = None
    for i, item in enumerate(tensor):
        if item != 0:
            if first_non_zero is None:
                first_non_zero = i
            last_non_zero = i+1
    return first_non_zero, last_non_zero

def crop_zeros(tensor1, tensor2):
    start, stop = find_padding_limits(tensor1)
    return tensor2[start:stop]

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

def get_preds(model, X_test, y_test):
    preds = []
    model.eval()
    for X, y in list(zip(X_test, y_test)):
        X_ = torch.cat([torch.zeros(5000),X,torch.zeros(5000)],dim=0)
        y_ = torch.cat([torch.zeros(5000),y,torch.zeros(5000)],dim=0)
        X_one_hot = one_hot(X_).T
        outputs = model(X_one_hot).view(-1)
        preds.append((X_.cpu().detach().numpy(), y_.cpu().detach().numpy(), outputs.cpu().detach().numpy()))
    return preds

def get_report(preds):
    report = dict()
    for idx, attrs in tqdm(enumerate(preds)):
        #trx = trxps[idx]
        seq, target, out = attrs[0], attrs[1], attrs[2]
        out, target, sequence = crop_zeros(seq, out), crop_zeros(seq, target), crop_zeros(seq, seq)
        out, target, sequence = torch.tensor(out), torch.tensor(target), torch.tensor(sequence)
        preds = bin_pred(out, 0.5)
        recall = recall_score(target, preds)
        iou = iou_score(target, preds)
        orfs_coord = pred_orfs(out.detach().numpy(), map_back(sequence), 7, 0.25)
        report[idx] = {'out': out,
                       'mapped_seq':sequence,
                       'mapped_cds': target,
                       'iou': iou,
                       'recall': recall,
                       'orfs_coord': orfs_coord}
    return report