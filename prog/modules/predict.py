from model import FOMOnet
from utils import *
import torch
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('input_path')
parser.add_argument('output_path')
args = parser.parse_args()

def parse_fasta(file_path):
    with open(file_path, 'r') as file:
        sequences = []
        header = None
        sequence = ''
        for line in file:
            if line.startswith('>'):
                if header:
                    sequences.append((header, sequence))
                    sequence = ''
                header = line.strip()
            else:
                sequence += line.strip()
        if header:
            sequences.append((header, sequence))
    return sequences

def get_preds(model, X_test):
    preds = []
    model.eval()
    for X in X_test:
        pad = torch.zeros(4,1000)
        X = torch.cat([pad,X,pad],dim=1).view(1,4,-1).cuda()
        out = model(X).view(-1)
        out = out[pad.shape[1]:-pad.shape[1]].cpu().detach()
        preds.append(out)
    return preds

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fomonet = FOMOnet.to(device)
    checkpoint = torch.load(args.model)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    fomonet.load_state_dict(state_dict)
    fomonet.eval()

    data = parse_fasta(args.input_path)

    for trx, seq in data:
        pad = torch.zeros(4,1000)
        X = torch.cat([pad,X,pad],dim=1).view(1,4,-1).cuda()
        out = fomonet(X).view(-1)
        out = out[pad.shape[1]:-pad.shape[1]].cpu().detach()


        

        