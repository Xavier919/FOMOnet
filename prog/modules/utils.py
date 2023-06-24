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
    mapping = {0:[0,0,0,0], 1:[1,0,0,0], 2:[0,1,0,0], 3:[0,0,1,0], 4:[0,0,0,1]}
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


def find_orfs(seq):
    start_codons, stop_codons = ['ATG'], ['TGA', 'TAA', 'TAG']
    frames = [0,1,2]
    ORFs = []
    #for start_codon in start_codons:
    for frame in frames:
        starts, stops = [], []
        for idx in list(range(frame, len(seq), 3)):
            codon = seq[idx:idx+3]
            #if codon == start_codon
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
                    #ORFs.append((start, stop, frame+1))
                    ORFs.append((start, stop))
                    break
    ORFs = sorted(ORFs, key=lambda x: x[0])
    return ORFs

def sliding_window(sequence, window_size):
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        if np.any(window > 0.5) and np.any(window < 0.5):
            yield i, i+window_size
def pred_orfs(out, seq, window_size):
    window_coordinates = list(sliding_window(out, window_size))
    orfs = find_orfs(seq)
    preds = []
    for start, stop in orfs:
        start_bool = any(start in range(x[0], x[1]) for x in window_coordinates)
        stop_bool = any(stop in range(x[0], x[1]+1) for x in window_coordinates)
        if start_bool==True and stop_bool==True:
            preds.append((start,stop))
    return preds

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