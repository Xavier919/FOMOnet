from tqdm import tqdm_notebook as tqdm
import torch
import gzip
import csv
from modules.utils import *

class Data:
    def __init__(self, alt_trx, alt_cds, alt_info):

        self.alt_trx = alt_trx
        self.alt_cds = alt_cds
        self.alt_info = alt_info
    
    def read_fasta(self, filename):
        if filename.endswith(".gz"):
            fp = gzip.open(filename, "rt")
        else:
            fp = open(filename, "r")
        name, seq = None, []
        for line in fp:
            line = line.rstrip()
            if line.startswith(">"):
                if name: yield (name, "".join(seq))
                name, seq = line, []
            else:
                seq.append(line)
        if name: yield (name, "".join(seq))
        fp.close()

    def get_trx(self):
        trx_dict = dict()
        for name, seq in tqdm(self.read_fasta(self.alt_trx)):
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

    def get_trxps_info(self):
        dataset = dict()
        with open(self.alt_info, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for n, row in tqdm(enumerate(reader)):
                if n==0:
                    cols = row
                    continue
                line = dict(zip(cols, row))
                trx = line['Transcript stable ID']
                dataset[trx] = {'gene_id': line['Gene stable ID'],
                                'gene_name': line['Gene name'],
                                'tsl': line['Transcript support level (TSL)'],
                                'biotype': line['Transcript type']}
        return dataset

    def get_dataset(self):
        trx_dict = self.get_trx()
        cds_dict = self.get_cds()
        info_dict = self.get_trxps_info()
        dataset = dict()
        for trx, attrs in tqdm(trx_dict.items()):
            trx_id = trx.split('|')[1]
            if info_dict[trx_id]['biotype'] != 'protein_coding' or info_dict[trx_id]['tsl'] not in ['tsl1', 'tsl2', 'tsl3']:
                continue
            trx_seq, seq_len = attrs['seq'], len(attrs['seq'])
            cds_seq = cds_dict[trx]['seq']
            if trx_id in dataset:
                seq_tensor = dataset[trx_id]['mapped_cds']
            else:
                seq_tensor = torch.zeros(1, seq_len).view(-1)
            if cds_seq == 'Sequence unavailable' or seq_len > 30000:
                continue
            start, stop = find_coordinates(cds_seq, trx_seq)
            seq_tensor = map_cds(seq_tensor, start, stop, 1)
            if 1 in seq_tensor:
                dataset[trx_id] = {'mapped_seq': map_seq(trx_seq),
                                   'mapped_cds': seq_tensor}
        return dataset, info_dict
