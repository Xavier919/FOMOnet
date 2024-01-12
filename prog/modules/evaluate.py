#imports
import torch
from torch import cat
import torch
#project specific imports
from model import FOMOnet
from utils import *

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
        pad = torch.zeros(4,1000)
        X = torch.cat([pad,X,pad],dim=1).view(1,4,-1).cuda()
        out = model(X).view(-1)
        out = out[pad.shape[1]:-pad.shape[1]].cpu().detach()
        preds.append(out)
    return preds

def get_xFOMO(model, X_test, y_test, trxps):
    model.eval()
    trx_xscores = dict()
    w_size = 7
    batch_size = 8
    for X,y,trx in zip(X_test,y_test,trxps):
        xscores = []
        masked_X = []
        pad = torch.zeros(4,1000)
        pad_length, X_length = pad.shape[-1], X.shape[-1]
        X = torch.cat([pad,X,pad],dim=1).view(4,-1)
        for i in range(pad_length,pad_length+X_length):
            X_ = X.clone().T
            X_[i:i+w_size] = torch.tensor([0.,0.,0.,0.])
            masked_X.append(X_.T)
        for i in range(0, len(masked_X), batch_size):
            batch = masked_X[i:i+batch_size]
            size = len(batch)
            batch = pad_seqs(batch, 4, min_pad=0).cuda()
            batch = batch.view(size, 4, -1)
            outputs = model(batch).view(size,1,-1)
            for out in outputs:
                out = out.flatten()
                out = out[pad_length:-pad_length].cpu().detach()
                pred = bin_pred(out, 0.5)
                iou = iou_score(y, pred)
                xscores.append(iou)
        trx_xscores[trx] = xscores
        print(len(trx_xscores))
    return trx_xscores

def get_orfs(preds, seqs_test, trxps):
    orfs = dict()
    for idx, out in enumerate(preds):
        trx = trxps[idx]
        seq_test = seqs_test[idx]
        orfs[trx] = orf_retrieval(seq_test, out.numpy(), t = 0.5, w_size = 10, cds_cov = 0.75)
    return orfs