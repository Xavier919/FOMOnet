from modules.evaluate import get_preds, get_orfs
from modules.model import FOMOnet
from modules.utils import map_seq
import pickle
import argparse
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('split')
parser.add_argument('model')
parser.add_argument('tag', type=str)
args = parser.parse_args()

if __name__ == "__main__":

    split = pickle.load(open(args.split, 'rb'))
    _, _, test, trxps = split

    seqs_test, y_test = test
    
    X_test = [map_seq(x) for x in seqs_test]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fomonet = FOMOnet().to(device)
    
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        fomonet = nn.DataParallel(fomonet)
    checkpoint = torch.load(args.model)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    fomonet.load_state_dict(state_dict)

    preds = get_preds(fomonet, X_test)

    orfs = get_orfs(preds, seqs_test, trxps)

    pickle.dump((preds, y_test), open(f'preds_{args.tag}.pkl', 'wb'))
    pickle.dump(orfs, open(f'orfs_{args.tag}.pkl', 'wb'))