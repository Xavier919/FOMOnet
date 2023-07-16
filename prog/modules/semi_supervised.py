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
parser.add_argument('Xy_train')
parser.add_argument('Xy_test')
parser.add_argument('Xy_alt')
parser.add_argument('trxps')
parser.add_argument('model')
parser.add_argument('kernel', type=int)
args = parser.parse_args()

def get_preds(model, X_test):
    preds = []
    model.eval()
    for X in X_test:
        pad = torch.zeros(4,5000)
        X_ = torch.cat([pad,X,pad],dim=1).view(1,4,-1)
        out = model(X_).view(-1)
        preds.append(out[pad.shape[1]:-pad.shape[1]].cpu().detach())
    return preds

def semi_supervised_dataset(preds, seqs, trxps):
    ss_dataset = dict()
    for idx, out in enumerate(preds):
        trx, seq = trxps[idx], seqs[idx]
        seq_tensor = torch.zeros(len(seq))
        coordinates = pred_orfs(out, seq, window_size=7, threshold=0.25)
        for start, stop in coordinates:
            seq_tensor[start:stop] = 1
        if 1 in seq_tensor:
            ss_dataset[trx] = {'mapped_seq': seq,
                               'mapped_cds': seq_tensor.view(1,-1)}
    return ss_dataset


if __name__ == "__main__":

    X_train, y_train = pickle.load(open(args.Xy_train, 'rb'))
    X_test, y_test = pickle.load(open(args.Xy_test, 'rb'))
    
    X_alt = pickle.load(open(args.Xy_alt, 'rb'))

    mapped_X_alt = [map_seq(x) for x in X_alt]

    trxps = pickle.load(open(args.trxps, 'rb'))

    fomonet = FOMOnet(k=args.kernel)
    fomonet.load_state_dict(torch.load(args.model, map_location=torch.device('cuda')))

    preds = get_preds(fomonet, mapped_X_alt)
    ss_dataset = semi_supervised_dataset(preds, X_alt, trxps)

    pickle.dump(preds, open('ss_preds.pkl', 'wb'))
    pickle.dump(ss_dataset, open('ss_dataset.pkl', 'wb'))

    ### new dataset
    X_alt = [x['mapped_seq'] for x in ss_dataset.values()]
    y_alt = [x['mapped_cds'] for x in ss_dataset.values()]

    X, y = X_train + X_alt, y_train + y_alt

    Xy = (X,y)

    pickle.dump(Xy, open('data/ss_Xy.pkl', 'wb'))
