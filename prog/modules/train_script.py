#imports
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import pickle
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
#project specific imports
from model import FOMOnet
from batch_sampler import BatchSampler
from transcripts import Transcripts
from utils import *
import argparse
import time


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

    split = pickle.load(open(args.split, 'rb'))
    train, test, trxps = split

    seqs_train, y_train = train
    seqs_test, y_test = test
    
    X_train, X_test = [map_seq(x) for x in seqs_train], [map_seq(x) for x in seqs_test]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    train_set = Transcripts(X_train, y_train)
    test_set = Transcripts(X_test, y_test)

    batch_size = args.batch_size
    epochs = args.epochs
    
    train_sampler = BatchSampler(train_set, batch_size)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=utility_fct, num_workers=8)

    test_sampler = BatchSampler(test_set, batch_size)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, collate_fn=utility_fct, num_workers=8)

    fomonet = FOMOnet(k=args.kernel, p=args.dropout).to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        fomonet = nn.DataParallel(fomonet) 
        
    optimizer = optim.Adam(fomonet.parameters(), args.lr)
    loss_function = nn.BCELoss(reduction='none').to(device) 

    print(f'tag:{args.tag}\n')
    print(f'learning rate:{args.lr}\n')
    print(f'batch size:{args.batch_size}\n')
    print(f'kernel:{args.kernel}\n')
    print(f'dropout:{args.dropout}\n')

    start_time = time.time()

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
            #loss = loss_function(outputs, y)
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
            #test_loss = loss_function(outputs, y)
            test_loss = test_loss.cpu().detach().numpy()
            test_writer.add_scalar("Loss/test", test_loss, epoch)
            test_losses.append(test_loss)
        if np.mean(test_losses) < best_model:
            best_model = np.mean(test_losses)
            torch.save(fomonet.state_dict(), f'fomonet{args.tag}.pt')
        print(f'{epoch}_{np.mean(test_losses)}')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time} seconds")

    writer.flush()
    test_writer.flush()
    writer.close()
    test_writer.close()