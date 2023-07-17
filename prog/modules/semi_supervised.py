#imports
import torch
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from torch import cat
from sklearn.metrics import auc
from sklearn.metrics import recall_score
import torch
#project specific imports
from model import FOMOnet
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('Xy_train')
parser.add_argument('alt_dataset')
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

def semi_supervised_dataset(preds, alt_dataset):
    ss_dataset = dict()
    for idx, out in enumerate(preds):
        trxps = [x for x in alt_dataset.keys()]
        seqs = [x['mapped_seq'] for x in alt_dataset.values()]
        orfs_lists = [x['orfs_list'] for x in alt_dataset.values()]
        trx, seq, orfs_list = trxps[idx], seqs[idx], orfs_lists[idx]
        seq_tensor = torch.zeros(len(seq))
        coordinates = pred_orfs(out.numpy(), seq, window_size=7, threshold=0.25)
        for tup in coordinates:
            if tup in orfs_list:
                seq_tensor[tup[0]:tup[1]] = 1
        if 1 in seq_tensor:
            ss_dataset[trx] = {'mapped_seq': seq,
                               'mapped_cds': seq_tensor.view(1,-1)}
    return ss_dataset


if __name__ == "__main__":

    X_train, y_train = pickle.load(open(args.Xy_train, 'rb'))
    
    alt_dataset = pickle.load(open(args.alt_dataset, 'rb'))

    mapped_X_alt = [map_seq(x['mapped_seq']) for x in alt_dataset.values()]

    fomonet = FOMOnet(k=args.kernel)
    fomonet.load_state_dict(torch.load(args.model, map_location=torch.device('cuda')))

    preds = get_preds(fomonet, mapped_X_alt)
    ss_dataset = semi_supervised_dataset(preds, alt_dataset)

    pickle.dump(preds, open('ss_preds.pkl', 'wb'))
    pickle.dump(ss_dataset, open('ss_dataset.pkl', 'wb'))

    ### new dataset

    X_alt = [x['mapped_seq'] for x in ss_dataset.values()]
    y_alt = [x['mapped_cds'] for x in ss_dataset.values()]

    X, y = X_train + X_alt, y_train + y_alt

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    train_split = (X_train,y_train)
    test_split = (X_test,y_test)
    pickle.dump(train_split, open('ss_Xy_train.pkl', 'wb'))
    pickle.dump(test_split, open('ss_Xy_test.pkl', 'wb'))