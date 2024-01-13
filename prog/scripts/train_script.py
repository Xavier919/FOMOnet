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
from modules.model import FOMOnet
from modules.batch_sampler import BatchSampler
from modules.transcripts import Transcripts
from modules.utils import get_loss, map_seq, utility_fct
import argparse
import time


writer = SummaryWriter()
valid_writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('split')
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('lr', type=float)
parser.add_argument('l2', type=int)
parser.add_argument('tag', type=str)
args = parser.parse_args()

if __name__ == "__main__":

    split = pickle.load(open(args.split, 'rb'))
    train, valid, _, trxps = split

    seqs_train, y_train = train
    seqs_valid, y_valid = valid
    
    X_train, X_valid = [map_seq(x) for x in seqs_train], [map_seq(x) for x in seqs_valid]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    train_set = Transcripts(X_train, y_train)
    valid_set = Transcripts(X_valid, y_valid)

    batch_size = args.batch_size
    epochs = args.epochs
    
    train_sampler = BatchSampler(train_set, batch_size)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=utility_fct, num_workers=8)

    valid_sampler = BatchSampler(valid_set, batch_size)
    valid_loader = DataLoader(valid_set, batch_sampler=valid_sampler, collate_fn=utility_fct, num_workers=8)

    fomonet = FOMOnet().to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        fomonet = nn.DataParallel(fomonet) 
        
    optimizer = optim.Adam(fomonet.parameters(), lr = args.lr, weight_decay=args.l2)
    loss_function = nn.BCELoss(reduction='none').to(device) 

    print(f'tag:{args.tag}\n')
    print(f'learning rate:{args.lr}\n')
    print(f'batch size:{args.batch_size}\n')
    print(f'l2:{args.l2}\n')

    start_time = time.time()

    best_model = 1.0
    early_stop_cnt = 0
    early_stop = 5
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
        valid_losses = []
        for X, y in valid_loader:
            size = len(X)
            X = X.view(size,4,-1).cuda()
            y = y.view(size,1,-1).cuda()
            outputs = fomonet(X).view(size,1,-1)
            valid_loss = get_loss(X, y, outputs, loss_function)
            valid_loss = valid_loss.cpu().detach().numpy()
            valid_writer.add_scalar("Loss/valid", valid_loss, epoch)
            valid_losses.append(valid_loss)
        print(f'{epoch}_{np.mean(valid_losses)}')

        if np.mean(valid_losses) < best_model:
            best_model = np.mean(valid_losses)
            torch.save(fomonet.state_dict(), f'fomonet{args.tag}.pt')
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        
        if early_stop_cnt == early_stop:
            print('early stop')
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time} seconds")

    writer.flush()
    valid_writer.flush()
    writer.close()
    valid_writer.close()