#imports
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt
from torch import cat
import torch
import torch.nn as nn
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

def get_preds(model, X_test):
    preds = []
    model.eval()
    for X in X_test:
        pad = torch.zeros(4,X.shape[-1]+100)
        X = torch.cat([pad,X,pad],dim=1).view(1,4,-1).cuda()
        out = model(X).view(-1)
        out = out[pad.shape[1]:-pad.shape[1]].cpu().detach()
        preds.append(out)
    return preds

#def get_report(preds, seqs_test, y_test, trxps):
#    report = dict()
#    outputs = dict()
#    for idx, out in enumerate(preds):
#        out = out[0]
#        trx = trxps[idx]
#        seq_test = seqs_test[idx]
#        target = y_test[idx].view(-1)
#        pred = bin_pred(out, 0.5)
#        recall = recall_score(target, pred)
#        iou = iou_score(target, pred)
#        report[trx] = {'pred_orfs': orf_retrieval(seq_test, out.numpy()),
#                       'iou': iou,
#                       'recall': recall}
#        outputs[trx] = {'out': out.numpy()}
#    return report, outputs

def get_orfs(preds, seqs_test, trxps):
    orfs = dict()
    for idx, out in enumerate(preds):
        trx = trxps[idx]
        seq_test = seqs_test[idx]
        orfs[trx] = orf_retrieval(seq_test, out.numpy())
    return orfs

if __name__ == "__main__":

    split = pickle.load(open(args.split, 'rb'))
    _, test, trxps = split

    seqs_test, y_test = test
    
    X_test = [map_seq(x) for x in seqs_test]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fomonet = FOMOnet(k=args.kernel).to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        fomonet = nn.DataParallel(fomonet)

    fomonet.load_state_dict(torch.load(args.model))

    preds = get_preds(fomonet, X_test)

    orfs = get_orfs(preds, seqs_test, trxps)

    pickle.dump((preds, y_test), open(f'preds_{args.tag}.pkl', 'wb'))
    pickle.dump(orfs, open(f'orfs_{args.tag}.pkl', 'wb'))