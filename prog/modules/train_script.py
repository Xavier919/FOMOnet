#imports
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import pickle
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np
import torch.nn as nn
#project specific imports
from model import FOMOnet
from transcripts import Transcripts
from utils import *
import argparse

#create runs directory for curves visualization with tensorboard
#shutil.rmtree(f'runs/', ignore_errors=True)
#os.mkdir('runs')
writer = SummaryWriter()
test_writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('split')
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('lr', type=float)
parser.add_argument('kernel', type=int)
parser.add_argument('dropout', type=float)
parser.add_argument('tag', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    print(f'tag:{args.tag}\n')
    print(f'learning rate:{args.lr}\n')
    print(f'batch size:{args.batch_size}\n')
    print(f'kernel:{args.kernel}\n')

    split = pickle.load(open(args.split, 'rb'))
    train, test, trxps = split

    X_train, y_train = train
    X_test, y_test = test

    #for synthtetic data
    y_train = y_train + y_train
    X_train = X_train + [n_mask(x, pct=15) for x in X_train]

    #convert to one-hot encoding
    X_train, X_test = [map_seq(x) for x in X_train], [map_seq(x) for x in X_test]

    #pre-processing data for pytorch DataLoader
    train_set = Transcripts(X_train, y_train)
    test_set = Transcripts(X_test, y_test)

    #hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs

    #create DataLoader object for train & test data
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=utility_fct, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=utility_fct, shuffle=True, num_workers=8)

    #instantiate model, optimizer and loss function
    fomonet = FOMOnet(k=args.kernel, p=args.dropout).cuda()

    optimizer = optim.Adam(fomonet.parameters(), args.lr)
    loss_function = nn.BCELoss(reduction='none').cuda()

    #train model
    best_model = 1.0
    for epoch in range(epochs):
        fomonet.train()
        losses = []
        for X, y in train_loader:
            size = len(X)
            X = X.view(size,4,-1).cuda()
            y = y.view(size,1,-1).cuda()
            outputs = fomonet(X).view(size,1,-1)
            fomonet.zero_grad()
            loss = get_loss(X, y, outputs, loss_function)
            loss.backward()
            optimizer.step()
            loss = loss.cpu().detach().numpy()
            writer.add_scalar("Loss/train", loss, epoch)
            losses.append(loss)
        print(f'{epoch}_{np.mean(losses)}')
        fomonet.eval()
        test_losses = []
        for X, y in test_loader:
            size = len(X)
            X = X.view(size,4,-1).cuda()
            y = y.view(size,1,-1).cuda()
            outputs = fomonet(X).view(size,1,-1)
            test_loss = get_loss(X, y, outputs, loss_function)
            test_loss = test_loss.cpu().detach().numpy()
            test_writer.add_scalar("Loss/test", test_loss, epoch)
            test_losses.append(test_loss)
        if np.mean(test_losses) < best_model:
            best_model = np.mean(test_losses)
            torch.save(fomonet.state_dict(), f'fomonet{args.tag}.pt')
        print(f'{epoch}_{np.mean(test_losses)}')

    writer.flush()
    test_writer.flush()
    writer.close()
    test_writer.close()