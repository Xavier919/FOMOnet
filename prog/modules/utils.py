import torch
import torch
import random
from torch.nn.utils.rnn import pad_sequence

def pack_seqs(Xy):
    seq1, seq2 = zip(*Xy)
    #seq1, seq2 = Xy
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

def translate(trxp_seq, frame):
    amino_acids = list()
    for idx in list(range(frame, len(trxp_seq), 3)):
        codon = trxp_seq[idx:idx+3]
        if len(codon) == 3 and codon in self.codon_table:
            amino_acids.append(self.codon_table[codon])
    return ''.join(amino_acids)

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

codon_table = {
            'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
            'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
            'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
            'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
            'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
            'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
            'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
            'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
            }

def shuffle_seq(seq, start, stop):
    start_codon, stop_codon = seq[start:start+3], seq[stop-3:stop]
    seq_copy_list = list(seq)  
    random.shuffle(seq_copy_list) 
    seq_copy_list[start:start+3] = list(start_codon)
    seq_copy_list[stop-3:stop] = list(stop_codon)
    syn_seq = ''.join(seq_copy_list) 
    # Check and replace start and stop codons between start and stop
    for i in range(start+3, stop-3, 3):
        codon = syn_seq[i:i+3]
        if codon in ['TAA', 'TAG', 'TGA', 'ATG']:
            new_codons = [x for x in codon_table if x not in ['TAA', 'TAG', 'TGA', 'ATG']]
            new_codon = random.choice(new_codons)
            seq_copy_list[i:i+3] = list(new_codon)
    syn_seq = ''.join(seq_copy_list)
    return syn_seq

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