import torch
import random
import numpy as np
from itertools import groupby
from operator import itemgetter

def pad_seqs(seqs, num_chan, min_pad=100):
    pad_seqs = []
    max_len = max([x.shape[1] for x in seqs])+min_pad
    #here, 5 is the number of maxpooling layers
    max_len = [i for i in range(0,30500) if i % (2^5) == 0 and i >= max_len][0]
    for seq in seqs:
        diff_len = max_len - seq.shape[1]
        padL, padR = torch.zeros(num_chan, diff_len//2), torch.zeros(num_chan, diff_len//2+diff_len%2)
        pad_seq = torch.cat([padL, seq, padR], dim=1)
        pad_seqs.append(pad_seq)
    return torch.stack(pad_seqs, dim=0)

def utility_fct(Xy):
    seq1, seq2 = zip(*Xy)
    X, y = pad_seqs(seq1, 4), pad_seqs(seq2, 1)
    return (X, y)

def get_loss(X, y, out, loss_fct):
    loss = loss_fct(out, y).cuda()
    zero_mask = torch.all(X == 0, dim=1)
    zero_mask = zero_mask.unsqueeze(1)
    zero_mask = zero_mask.expand(-1, 1, -1)
    loss[zero_mask] = 0.
    lens = torch.sum(X, dim=(1,-1))
    loss_sums = torch.sum(loss, dim=(1,-1))
    return (loss_sums/lens).mean()

def map_seq(seq):
    mapping = {'N':[0.,0.,0.,0.], 'A':[1.,0.,0.,0.], 'T':[0.,1.,0.,0.], 'G':[0.,0.,1.,0.], 'C':[0.,0.,0.,1.]}
    return torch.tensor([mapping[x] for x in seq]).T

def find_orfs(seq, long=True, nc=False):
    start_codons, stop_codons = ['ATG'], ['TGA', 'TAA', 'TAG']
    if nc: start_codons = ['ATG', 'TTG', 'GTG', 'CTG']
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
                    if long == True:
                        break
    orfs = sorted(orfs, key=lambda x: x[0])
    return orfs

def check_drop(w, t, edge):
    if edge:
        return np.all(w) >= t
    else:
        return t <= np.max(w)-np.min(w)

def get_window(out, idx, w_size):
    edge = False
    if idx < w_size:
        edge = True
        return edge, out[:idx+w_size+3]
    elif idx+w_size+3 > len(out):
        edge = True
        return edge, out[idx-w_size:]
    else:
        return edge, out[idx-w_size:idx+w_size+3]

def valid_start(start, stops, idx):
    return any(i > start for i in stops[idx+1:])

def orf_retrieval(seq, out, t = 0.5, w_size = 7, cds_cov = 0.75):
    start_codons, stop_codons = ['ATG','TTG','GTG','CTG'], ['TGA','TAG','TAA']
    cds = []
    seq_len = len(seq)
    for frame in range(3):
        stops = [i for i in range(frame, seq_len, 3) if seq[i:i+3] in stop_codons][::-1]
        for idx, stop in enumerate(stops):
            e, w = get_window(out, stop, w_size)
            starts = [i for i in range(stop-3,-1,-3) if seq[i:i+3] in start_codons]
            if len(starts) == 0 or not check_drop(w, t, e):
                continue
            best_codon, best_codon_idx, best_cds_cov = None, None, 0
            for start in starts:
                e, w = get_window(out, start, w_size)
                if valid_start(start, stops, idx) or not check_drop(w[::-1], t, e) or stop - start < 90:
                    continue
                cov = np.sum(out[start:stop] >= t)/(stop-start)
                if best_codon == None or cov > best_cds_cov:
                    best_codon, best_codon_idx, best_cds_cov = seq[start:start+3], start, cov
            if best_codon != None and best_cds_cov >= cds_cov:
                cds.append((best_codon_idx, stop+3))
    return cds

def build_fasta(data, filename):
    trx_seqs = []
    for trx, seq in data:
        header = f'>{trx}'
        trx_seq = header + '\n' + seq.upper()
        trx_seqs.append(trx_seq)
    fasta_text = '\n'.join(trx_seqs)
    with open(filename + '.fa', 'w') as fasta_file:
        fasta_file.write(fasta_text)

def xfomo(iou_list, seq, cds_start, cds_stop, min_motif_len=10):
    median = np.median(iou_list)
    std = np.std(iou_list)
    seq_len = len(seq)
    idx_arr = np.where(iou_list < median-(2*std))[0]
    grps_idx = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(idx_arr), lambda x: x[0]-x[1])]
    results = dict()
    for grp in grps_idx:
        low_bnd, high_bnd = grp[0]-3, grp[-1]+3
        motif = seq[low_bnd:high_bnd+1]
        if len(motif) < min_motif_len:
            continue
        start_dist, stop_dist = low_bnd-cds_start, low_bnd-cds_stop
        if motif not in results:
            results[motif] = {'start_dist': [start_dist],
                              'stop_dist': [stop_dist],
                              'trx_loc': [low_bnd/seq_len]} 
        else:
            results[motif]['start_dist'].append(start_dist)
            results[motif]['stop_dist'].append(stop_dist)
            results[motif]['trx_loc'].append(low_bnd/seq_len)
    return results