import torch
import random
import numpy as np

def pad_seqs(seqs, num_chan):
    pad_seqs = []
    max_len = max([x.shape[1] for x in seqs])+500
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

def find_cds_coordinates(trx, cds):
    start = trx.find(cds)
    stop = start + len(cds)
    return start, stop

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

def n_mask(input_string, seed=42, pct=15):
    random.seed(seed)
    n_count = int(len(input_string) * pct / 100)
    n_indices = random.sample(range(len(input_string)), n_count)
    return ''.join(['N' if i in n_indices else char for i, char in enumerate(input_string)])

def find_frame(start, stop):
    valid = False
    for i in range(3):
        if (start - i) % 3 == 0 and (stop - i) % 3 == 0:
            valid = True
    return valid

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

def check_drop(w, t):
    return t <= np.max(w)-np.min(w)

def get_window(out, idx, w_size):
    if idx < w_size:
        return out[:idx+w_size+3]
    elif idx+w_size+3 > len(out):
        return out[idx-w_size:]
    else:
        return out[idx-w_size:idx+w_size+3]

def valid_start(start, stops, idx):
    return any(i > start for i in stops[idx+1:])

def orf_retrieval(seq, out, t = 0.5, w_size = 10):
    start_codons, stop_codons = ['ATG','CTG','GTG','TTG'], ['TGA','TAG','TAA']
    cds = []
    seq_len = len(seq)
    for frame in range(3):
        stops = [i for i in range(frame, seq_len, 3) if seq[i:i+3] in stop_codons][::-1]
        for idx, stop in enumerate(stops):
            w = get_window(out, stop, w_size)
            starts = [i for i in range(stop-3,-1,-3) if seq[i:i+3] in start_codons]
            if len(starts) == 0 or not check_drop(w, t):
                continue
            best_codon, best_codon_idx = None, None
            for start in starts:
                w = get_window(out, start, w_size)[::-1]
                if valid_start(start, stops, idx) or not check_drop(w, t) or stop - start < 90:
                    continue
                if best_codon == None or best_codon_idx < start:
                    best_codon, best_codon_idx = seq[start:start+3], start
            if best_codon != None:
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