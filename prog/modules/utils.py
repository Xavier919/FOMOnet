import torch
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def pack_seqs(Xy):
    seq1, seq2 = zip(*Xy)
    max_len = max([x.shape[-1] for x in seq1])+500
    seq1 = [torch.nn.functional.pad(x, pad=((max_len - x.shape[-1])//2, max_len - x.shape[-1] - (max_len - x.shape[-1])//2), mode='constant', value=0) for x in seq1]
    seq2 = [torch.nn.functional.pad(y, pad=((max_len - y.shape[-1])//2, max_len - y.shape[-1] - (max_len - y.shape[-1])//2), mode='constant', value=0) for y in seq2]
    X = pad_sequence([x for x in seq1], batch_first=True, padding_value=0)
    y = pad_sequence([y for y in seq2], batch_first=True, padding_value=0)
    return (X, y, one_hot(X))

def get_loss(outputs, X, y, loss_function):
    loss = loss_function(outputs, y)
    loss_mask = X != 0
    loss_masked = loss.where(loss_mask.view(len(X),-1), torch.tensor(0.).cuda())
    mean_loss = (loss_masked.sum(axis=1)/loss_mask.sum(axis=2).view(-1)).mean()
    return mean_loss

def one_hot(seqs):
    tensors = []
    mapping = {0:[0,0,0], 1:[1,0,0], 2:[0,1,0], 3:[0,0,1], 4:[0,0,0]}
    for seq in seqs:
        one_hot_vector = torch.tensor([mapping[int(val)] for val in seq.view(-1)])
        tensors.append(one_hot_vector.T)
    return torch.stack(tensors).float()

def map_seq(seq):
    mapping = dict(zip("NATGC", range(0,5)))
    return torch.Tensor([mapping[nt] for nt in seq])

def map_cds(seq_tensor, start, stop, num):
    ORF_loc = range(start, stop)
    for pos, _ in enumerate(seq_tensor):
        if pos in ORF_loc:
            seq_tensor[pos] = num
    return seq_tensor

def map_back(seq):
    mapping = dict(zip("NATGC", range(0,5)))
    mapping_back = {y:x for x,y in mapping.items()}
    return "".join([mapping_back[nt] for nt in seq.int().tolist() if nt != 0])


def find_orfs(seq, keep_longest=False, nc_starts=False):
    if nc_starts:
        start_codons, stop_codons = ['ATG', 'TTG', 'GTG', 'CTG'], ['TGA', 'TAA', 'TAG']
    else:
        start_codons, stop_codons = ['ATG'], ['TGA', 'TAA', 'TAG']
    frames = [0,1,2]
    orfs = []
    for frame in frames:
        starts, stops = [], []
        for idx in list(range(frame, len(seq), 3)):
            codon = seq[idx:idx+3]
            if codon in start_codons: 
                starts.append(idx)
            elif codon in stop_codons:
                stops.append(idx+3)
        stops = stops[::-1]
        for idx, stop in enumerate(stops):
            for start in starts:
                if stop - start < 90 or any(i > start for i in stops[idx+1:]):
                    continue
                else:
                    orfs.append((start, stop))
                    if keep_longest == True:
                        break
    orfs = sorted(orfs, key=lambda x: x[0])
    return orfs

def pred_orfs(out, seq, window_size=7, threshold=0.5):
    pred_orfs = []
    ws = window_size
    for start, stop in find_orfs(seq):
        if start < ws:
            start_window = out[:start+ws]
        else:
            start_window = out[start-ws:start+ws]
        min_val, max_val = np.min(start_window), np.max(start_window)
        min_index = np.unravel_index(np.argmin(start_window), start_window.shape)
        max_index = np.unravel_index(np.argmax(start_window), start_window.shape)
        if (max_val - min_val >= threshold and max_index > min_index) or (start < ws and any(i >= 0.5 for i in start_window)):
            if len(seq) < stop+ws:
                stop_window = out[stop-ws:]
            else:
                stop_window = out[stop-ws:stop+ws]
            min_val, max_val = np.min(stop_window), np.max(stop_window)
            min_index = np.unravel_index(np.argmin(stop_window), stop_window.shape)
            max_index = np.unravel_index(np.argmax(stop_window), stop_window.shape)
            if (max_val - min_val >= threshold and max_index < min_index) or (len(seq) < stop+ws and any(i >= 0.5 for i in stop_window)):
                pred_orfs.append((start, stop))
    return pred_orfs

def find_coordinates(orf_seq, trx_seq):
    length = len(orf_seq)
    start = trx_seq.find(orf_seq)
    stop = start+length
    return start, stop

def find_cds(y):
    in_CDS = []
    for target in enumerate(y.int()):
        if target[1] == 1:
            in_CDS.append(target[0])
    coordinates = (0, 0)
    if len(in_CDS) != 0:
        coordinates = (in_CDS[0], in_CDS[-1]+1)
    return coordinates