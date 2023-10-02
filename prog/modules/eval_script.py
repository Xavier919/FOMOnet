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
        pad = torch.zeros(4,X.shape[-1]//2)
        X = torch.cat([pad,X,pad],dim=1).view(1,4,-1).cuda()
        out = model(X).view(-1)
        out = out[pad.shape[1]:-pad.shape[1]].cpu().detach()
        preds.append(out)
    return preds


#def get_mask_iou(model, X_test, y_test):
#    iou_lists = []
#    model.eval()
#    w_size = 5
#    for X,y in zip(X_test,y_test):
#        iou_list = []
#        pad = torch.zeros(4,X.shape[-1]//2)
#        for i in range(0,X.shape[-1]):
#            X_ = X.clone().T
#            X_[i:i+w_size] = torch.tensor([0.,0.,0.,0.])
#            X_ = X_.T
#            X_ = torch.cat([pad,X_,pad],dim=1).view(1,4,-1).cuda()
#            out = model(X_).view(-1)
#            out = out[pad.shape[1]:-pad.shape[1]].cpu().detach()
#            pred = bin_pred(out, 0.5)
#            iou = iou_score(y, pred)
#            iou_list.append(iou)
#        iou_lists.append(iou_list)
#    return iou_lists

def get_xFOMO(model, X_test, y_test):
    model.eval()
    list_xscores = []
    w_size = 5
    batch_size = 8
    for X,y in zip(X_test,y_test):
        xscores = []
        masked_X = []
        pad = torch.zeros(4,X.shape[-1]//2)
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
        list_xscores.append(xscores)
        print(len(list_xscores))
    return list_xscores

def get_orfs(preds, seqs_test, trxps):
    orfs = dict()
    for idx, out in enumerate(preds):
        trx = trxps[idx]
        seq_test = seqs_test[idx]
        orfs[trx] = orf_retrieval(seq_test, out.numpy(), t = 0.25, w_size = 7, cds_cov = 0.75)
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
    checkpoint = torch.load(args.model)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    fomonet.load_state_dict(state_dict)


    #preds = get_preds(fomonet, X_test)

    #orfs = get_orfs(preds, seqs_test, trxps)

    #pickle.dump((preds, y_test), open(f'preds_{args.tag}.pkl', 'wb'))
    #pickle.dump(orfs, open(f'orfs_{args.tag}.pkl', 'wb'))

    xFOMO = get_xFOMO(fomonet, X_test, y_test)

    pickle.dump(xFOMO, open(f'xFOMO{args.tag}.pkl', 'wb'))