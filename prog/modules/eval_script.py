#imports
import torch
import pickle
from sklearn.model_selection import train_test_split
#project specific imports
from model_ import FOMOnet
from transcripts import Transcripts
from utils import *
from evaluate import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data')
parser.add_argument('model')
parser.add_argument('kernel', type=int)
parser.add_argument('dropout', type=float)
parser.add_argument('trxps', type=str)
parser.add_argument('tag', type=str)
args = parser.parse_args()

if __name__ == "__main__":

    X_test, y_test = pickle.load(open(args.data, 'rb'))

    trxps = pickle.load(open(args.trxps, 'rb'))
    _, trxps, _, _ = train_test_split(trxps, trxps, test_size=0.1, random_state=42)

    fomonet = FOMOnet(p=args.dropout, k=args.kernel)
    fomonet.load_state_dict(torch.load(args.model, map_location=torch.device('cuda')))

    preds = get_preds(fomonet, X_test)

    report = get_report(preds, y_test, trxps)

    pickle.dump(preds, open(f'preds_{args.tag}.pkl', 'wb'))
    pickle.dump(report, open(f'report_{args.tag}.pkl', 'wb'))