from tqdm import tqdm_notebook as tqdm
import torch
from modules.utils import *

class Data:
    def __init__(self, alt_trx, alt_cds):

        self.alt_trx = alt_trx
        self.alt_cds = alt_cds
    
    def get_trx(self):
        trx_dict = dict()
        for name, seq in tqdm(self.alt_trx):
            trx = name.split('>')[1]
            seq = str(seq)
            trx_dict[trx] = {'seq':seq}
        return trx_dict
    
    def get_cds(self):
        cds_dict = dict()
        for name, seq in tqdm(self.read_fasta(self.alt_cds)):
            trx = name.split('>')[1]
            seq = str(seq)
            cds_dict[trx] = {'seq':seq}
        return cds_dict

    def get_dataset(self):
        trx_dict = self.get_trx()
        cds_dict = self.get_cds()
        dataset = dict()
        for trx, attrs in tqdm(trx_dict.items()):
            trx_seq, seq_len = attrs['seq'], len(attrs['seq'])
            cds_seq = cds_dict[trx]['seq']
            seq_tensor = torch.zeros(1, seq_len).view(-1)
            if cds_seq != 'Sequence unavailable':
                start, stop = find_coordinates(cds_seq, trx_seq)
                seq_tensor = map_cds(seq_tensor, start, stop, 1)
            if 1 in seq_tensor:
                dataset[trx] = {'mapped_seq': map_seq(trx_seq),
                                'mapped_cds': seq_tensor}
        return dataset