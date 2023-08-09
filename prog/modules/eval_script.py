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
parser.add_argument('split')
parser.add_argument('model')
parser.add_argument('kernel', type=int)
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

def check_delta(out, t, edge):
    if edge and np.all(out) >= t:
        return True
    if len(out) == 0:
        return False
    max_ = out[0]
    for val in out[1:]:
        if val <= max_ - t:
            return True
        max_ = max(max_, val)
    return False

def check_drop(w, t):
    return t <= np.max(w)-np.min(w)

def get_window(out, idx, w_size):
    return out[idx-w_size:idx+w_size+3]

def valid_start(start, stops, idx):
    return any(i > start for i in stops[idx+1:])

def orf_retrieval(seq, out, t = 0.25, w_size = 7):
    start_codons, stop_codons = ['ATG','CTG','GTG','TTG'], ['TGA','TAG','TAA']
    cds = []
    seq_len = len(seq)
    for frame in range(3):
        stops = [i for i in range(frame, seq_len, 3) if seq[i:i+3] in stop_codons][::-1]
        for idx, stop in enumerate(stops):
            w = get_window(out, stop, w_size)
            if len(w) == 0:
                continue
            starts = [i for i in range(stop-3,-1,-3) if seq[i:i+3] in start_codons]
            if not check_drop(w, t):
                continue
            best_codon, best_codon_idx = None, None
            for start in starts:
                w = get_window(out, start, w_size)[::-1]
                if len(w) == 0:
                    continue
                if valid_start(start, stops, idx) or not check_drop(w, t):
                    continue
                if best_codon == None or best_codon_idx < start:
                    best_codon, best_codon_idx = seq[start:start+3], start
            if best_codon != None:
                cds.append((best_codon_idx, stop+3))
    return cds

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

    split = pickle.load(open(args.split, 'rb'))
    train, test, trxps = split

    X_train, y_train = train
    X_test, y_test = test
    
    X_train, X_test = [map_seq(x) for x in X_train], [map_seq(x) for x in X_test]

    fomonet = FOMOnet(k=args.kernel)
    fomonet.load_state_dict(torch.load(args.model, map_location=torch.device('cuda')))

    preds = get_preds(fomonet, X_test)
    report = get_report(preds, y_test, trxps)
    pickle.dump(report, open(f'report_{args.tag}.pkl', 'wb'))