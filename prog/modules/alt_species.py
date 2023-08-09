from tqdm import tqdm_notebook as tqdm
import torch
import gzip
import csv
from modules.utils import *

class Data:
    def __init__(self, trx, cds, info):
        self.biotype_grouping = {
            'protein_coding': 'protein_coding',
            'processed_transcript': 'processed_transcript',
            'miRNA': 'processed_transcript',
            'misc_RNA': 'processed_transcript',
            'unprocessed_pseudogene': 'pseudogene',
            'antisense': 'processed_transcript',
            'retained_intron': 'processed_transcript',
            'processed_pseudogene': 'pseudogene',
            'rRNA_pseudogene': 'pseudogene',
            'sense_intronic': 'processed_transcript',
            'lincRNA': 'processed_transcript',
            'snoRNA': 'processed_transcript',
            'snRNA': 'processed_transcript',
            'transcribed_unprocessed_pseudogene': 'pseudogene',
            'translated_processed_pseudogene': 'pseudogene',
            'nonsense_mediated_decay': 'nmd',
            'polymorphic_pseudogene': 'pseudogene',
            'transcribed_processed_pseudogene': 'pseudogene',
            'TEC': 'others',
            'sense_overlapping': 'processed_transcript',
            'TR_J_gene': 'others',
            'IG_J_gene': 'others',
            'IG_V_pseudogene': 'pseudogene',
            'TR_V_gene': 'others',
            'rRNA': 'processed_transcript',
            'TR_C_gene': 'others',
            'scaRNA': 'processed_transcript',
            'IG_V_gene': 'others',
            'pseudogene': 'pseudogene',
            'bidirectional_promoter_lncRNA': 'processed_transcript',
            'TR_V_pseudogene': 'pseudogene',
            'Mt_tRNA': 'processed_transcript',
            'unitary_pseudogene': 'pseudogene',
            'IG_C_gene': 'others',
            'IG_pseudogene': 'pseudogene',
            'transcribed_unitary_pseudogene': 'pseudogene',
            'IG_C_pseudogene': 'pseudogene',
            'IG_D_gene': 'others',
            'non_coding': 'processed_transcript',
            'ribozyme': 'processed_transcript',
            '3prime_overlapping_ncRNA': 'processed_transcript',
            'TR_J_pseudogene': 'pseudogene',
            'IG_J_pseudogene': 'pseudogene',
            'non_stop_decay': 'nmd',
            'sRNA': 'processed_transcript',
            'TR_D_gene': 'others',
            'scRNA': 'processed_transcript',
            'vaultRNA': 'processed_transcript',
            'Mt_rRNA': 'processed_transcript',
            'macro_lncRNA': 'processed_transcript',
            'lincrna': 'processed_transcript',
            'lncRNA': 'processed_transcript',
            'protein_coding_CDS_not_defined': 'others',
            'protein_coding_LoF': 'others',
            'vault_RNA': 'processed_transcript',
            'artifact': 'others',
            'translated_unprocessed_pseudogene': 'pseudogene'
            }

        self.trx = trx
        self.cds = cds
        self.info = info
    
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
        for name, seq in tqdm(self.read_fasta(self.trx)):
            trx = name.split('>')[1]
            seq = str(seq)
            trx_dict[trx.split('|')[1]] = {'seq':seq}
        return trx_dict
    
    def get_cds(self):
        cds_dict = dict()
        for name, seq in tqdm(self.read_fasta(self.cds)):
            trx = name.split('>')[1]
            seq = str(seq)
            cds_dict[trx.split('|')[1]] = {'seq':seq}
        return cds_dict

    def get_trx_info(self):
        dataset = dict()
        with open(self.info, 'r') as f:
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
                                'biotype': self.biotype_grouping[line['Transcript type']]}
        return dataset

    def get_dataset(self):
        trx_dict = self.get_trx()
        cds_dict = self.get_cds()
        info_dict = self.get_trx_info()
        dataset = dict()
        for trx, attrs in tqdm(trx_dict.items()):
            trx_seq, trx_len = attrs['seq'], len(attrs['seq'])
            cds_seq, biotype = cds_dict[trx]['seq'], info_dict[trx]['biotype']
            seq_tensor = torch.zeros(trx_len)
            if cds_seq == 'Sequence unavailable' or trx_len > 30000 or biotype == 'nmd':
                continue
            start, stop = find_cds_coordinates(cds_seq, trx_seq)
            if find_frame(start,stop) != True:
                continue
            seq_tensor[start:stop] = 1
            dataset[trx] = {'mapped_seq': trx_seq,
                            'mapped_cds': seq_tensor}
        return dataset, info_dict, trx_dict, cds_dict
