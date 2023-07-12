#imports
import random
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
#project specific imports
from build_dataset import *
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('ensembl_trx')
parser.add_argument('trx_orfs')
args = parser.parse_args()

if __name__ == "__main__":

    ensembl_trx = pickle.load(open(args.ensembl_trx, 'rb'))
    trx_orfs = pickle.load(open(args.trx_orfs, 'rb'))

    dataset = data.dataset(ensembl_trx, trx_orfs)

    X = [x['mapped_seq'] for x in dataset.values()]
    y = [x['mapped_cds'] for x in dataset.values()]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    train_split = (X_train,y_train)
    test_split = (X_test,y_test)
    pickle.dump(train_split, open('data/train_split.pkl', 'wb'))
    pickle.dump(test_split, open('data/test_split.pkl', 'wb'))