#imports
import torch
import torch.optim as optim
from torch import nn
import pickle
import numpy as np
#project specific imports
from model import FOMOnet
from transcripts import Transcripts
from utils import *
from evaluate import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data')
parser.add_argument('model')
parser.add_argument('tag', type=str)
args = parser.parse_args()

if __name__ == "__main__":

    X_test, y_test = pickle.load(open(args.data, 'rb'))

    #instantiate model
    fomonet = FOMOnet(num_channels=4)
    fomonet.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    preds = get_preds(fomonet, X_test, y_test)
    pickle.dump(preds, open(f'preds_{args.tag}.pkl', 'wb'))