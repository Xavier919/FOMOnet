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
#project specific imports
from model_ import FOMOnet
from transcripts import Transcripts
from utils import *
import argparse

#create runs directory for curves visualization with tensorboard
#shutil.rmtree(f'runs/', ignore_errors=True)
#os.mkdir('runs')
writer = SummaryWriter()
test_writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('train_split')
parser.add_argument('test_split')
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('lr', type=float)
parser.add_argument('wd', type=float)
parser.add_argument('dropout', type=float)
parser.add_argument('kernel', type=int)
parser.add_argument('tag', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    print(f'tag:{args.tag}\n')
    print(f'learning rate:{args.lr}\n')
    print(f'weight decay:{args.wd}\n')
    print(f'batch size:{args.batch_size}\n')
    print(f'dropout:{args.dropout}\n')
    print(f'kernel:{args.kernel}\n')

    #load train & test data
    X_train, y_train = pickle.load(open(args.train_split, 'rb'))
    X_test, y_test = pickle.load(open(args.test_split, 'rb'))

    #pre-processing data for pytorch DataLoader
    train_set = Transcripts(X_train, y_train)
    test_set = Transcripts(X_test, y_test)

    #hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs

    #create DataLoader object for train & test data
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=pack_seqs, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=pack_seqs, shuffle=True, num_workers=16)

    #instantiate model, optimizer and loss function
    fomonet = FOMOnet(p=args.dropout, k=args.kernel).cuda()

    optimizer = optim.Adam(fomonet.parameters(), args.lr, weight_decay=args.wd)
    loss_function = nn.BCELoss(reduction='none').cuda()

    #train model
    best_model = 1.0
    for epoch in range(epochs):
        fomonet.train()
        losses = []
        for batch in train_loader:
            X = batch[0].view(len(batch[0]),1,-1).cuda()
            y = batch[1].view(len(batch[1]),-1).cuda()
            X_one_hot = batch[2].view(len(batch[0]),4,-1).cuda()
            outputs = fomonet(X_one_hot).view(len(batch[0]),-1)
            fomonet.zero_grad()
            loss = get_loss(outputs, X, y, loss_function)
            loss.backward()
            optimizer.step()
            loss = loss.cpu().detach().numpy()
            writer.add_scalar("Loss/train", loss, epoch)
            losses.append(loss)
        print(f'{epoch}_{np.mean(losses)}')
        fomonet.eval()
        test_losses = []
        for batch in test_loader:
            X = batch[0].view(len(batch[0]),1,-1).cuda()
            y = batch[1].view(len(batch[1]),-1).cuda()
            X_one_hot = batch[2].view(len(batch[0]),4,-1).cuda()
            outputs = fomonet(X_one_hot).view(len(batch[0]),-1)
            test_loss = get_loss(outputs, X, y, loss_function)
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