#imports
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch import nn
import pickle
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np
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
parser.add_argument('train_data')
parser.add_argument('test_data')
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('lr', type=float)
parser.add_argument('tag', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    print(args.tag)

    #load train & test data
    X_train, y_train = pickle.load(open(args.train_data, 'rb'))
    X_test, y_test = pickle.load(open(args.test_data, 'rb'))

    X_train_L, y_train_L = [x for x in X_train if len(x) >= 2500 and len(x) < 15000], [x for x in y_train if len(x) >= 2500 and len(x) < 15000]
    X_train_S, y_train_S = [x for x in X_train if len(x) < 2500 and len(x) < 15000], [x for x in y_train if len(x) < 2500 and len(x) < 15000]

    #pre-processing data for pytorch DataLoader
    train_set_L = Transcripts(X_train_L, y_train_L)
    train_set_S = Transcripts(X_train_S, y_train_S)
    test_set = Transcripts(X_test, y_test)

    #hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    #create DataLoader object for train & test data
    train_loader_L = DataLoader(train_set_L, batch_size=batch_size, collate_fn=pack_seqs, shuffle=True, num_workers=8)
    train_loader_S = DataLoader(train_set_S, batch_size=batch_size, collate_fn=pack_seqs, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=pack_seqs, shuffle=True, num_workers=8)

    #instantiate model, optimizer and loss function
    fomonet = FOMOnet(num_channels=4).cuda()
    optimizer = optim.Adam(fomonet.parameters(), lr)
    loss_function = nn.BCELoss(reduction='none').cuda()

    #train model
    best_model = 1.0
    for epoch in range(epochs):
        fomonet.train()
        for batch in train_loader_L:
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
        S_losses = []
        for batch in train_loader_S:
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
            S_losses.append(loss)
            if loss < best_model:
                best_model = loss
                torch.save(fomonet.state_dict(), f'fomonet{args.tag}.pt')
        print(f'{epoch}_{np.mean(S_losses)}')
        fomonet.eval()
        #test_losses = []
        for batch in test_loader:
            X = batch[0].view(len(batch[0]),1,-1).cuda()
            y = batch[1].view(len(batch[1]),-1).cuda()
            X_one_hot = batch[2].view(len(batch[0]),4,-1).cuda()
            outputs = fomonet(X_one_hot).view(len(batch[0]),-1)
            test_loss = get_loss(outputs, X, y, loss_function)
            test_loss = test_loss.cpu().detach().numpy()
            #test_losses.append(test_loss)
            test_writer.add_scalar("Loss/test", test_loss, epoch)
        #print(f'{epoch}_{np.mean(test_losses)}')
        #if np.mean(test_losses) < best_model:
            #best_model = np.mean(test_losses)
            #torch.save(fomonet.state_dict(), f'fomonet{args.tag}.pt')

    writer.flush()
    test_writer.flush()
    writer.close()
    test_writer.close()