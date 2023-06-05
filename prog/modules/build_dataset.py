import csv
from Bio import SeqIO
import pyfaidx
import pickle
import numpy as np
import torch
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import random
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import random
import pandas as pd
from modules.utils import *

class Data:
    def __init__(self, OP_tsv, Ens_trx, trx_fasta, sorfs, unique_pept):
        
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
            }
        
        self.codon_table = {
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

        self.OP_tsv = OP_tsv
        self.Ens_trx = Ens_trx
        self.ensembl95_trxps = pyfaidx.Fasta(trx_fasta)
        self.OP_prot_MS, self.OP_trx_altprot = self.get_altprot_info()
        self.sorfs = sorfs
        self.unique_pept = unique_pept

    def get_op_trx(self):
        op_trx_accession = set()
        with open(self.OP_tsv, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for n, row in enumerate(reader):
                if n==0: 
                    continue
                if n==1:
                    cols = row
                    continue
                line = dict(zip(cols, row)) 
                if "ENST" in line['transcript accession'] :
                    line['transcript accession'] = line['transcript accession'].split('.')[0]
                    op_trx_accession.add(line['transcript accession'])
        return op_trx_accession

    def get_altprot_info(self):
        op_orf_ev = dict()
        op_trx_orfs = dict()
        with open(self.OP_tsv, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for n, row in enumerate(reader):
                if n == 0:
                    continue
                if n == 1:
                    cols = row
                    continue
                line = dict(zip(cols, row))
                
                if "ENST" not in line["transcript accession"]:
                    continue
                if not any( x in line["protein accession numbers"] for x in ["IP_", "II_", "ENSP"]):
                    continue 
                trx = line["transcript accession"].split(".")[0]
                if trx not in op_trx_orfs:
                    op_trx_orfs[trx] = [line["protein accession numbers"]]
                else:
                    op_trx_orfs[trx].append(line["protein accession numbers"])
                    
                op_orf_ev[line["protein accession numbers"]] = {
                    'MS': int(line["MS score"]), 
                    'TE': int(line["TE score"]),
                    }
            return op_orf_ev, op_trx_orfs

    def ensembl_trx(self):
        op_trx = self.get_op_trx()
        ensembl_trx = dict()
        with open(self.Ens_trx, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for n, row in tqdm(enumerate(reader)):
                if n==0:
                    cols = row
                    continue
                line = dict(zip(cols, row))
                trx = line['Transcript stable ID']
                sequence = str(self.ensembl95_trxps["|".join([line['Gene stable ID'], line['Transcript stable ID']])])
                orf_accessions = []
                if trx in op_trx:
                    if trx in self.OP_trx_altprot:
                        orf_accessions = self.OP_trx_altprot[trx]

                ensembl_trx[trx] = {
                    'gene_id':line["Gene stable ID"],
                    'tsl': line["Transcript support level (TSL)"],
                    'gene_name': line["Gene name"],
                    'biotype': self.biotype_grouping[line["Transcript type"]],
                    'og_biotype': line["Transcript type"],
                    'orf_accessions': orf_accessions,
                    'sequence': sequence
                }
        return ensembl_trx

    def trx_orfs(self, ensembl_trx):
        trx_orfs = dict()
        with open(self.OP_tsv, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for n, row in enumerate(reader):
                if n == 0:
                    continue
                if n == 1:
                    cols = row
                    continue
                line = dict(zip(cols, row))
                if 'ENST' not in line["transcript accession"]:
                    continue
                if not any(x in line["protein accession numbers"] for x in ["IP_", 'ENSP', "II_"]):
                    continue
                trx = line["transcript accession"].split(".")[0]
                altprot = line["protein accession numbers"].split(".")[0]
                seq = ensembl_trx[trx]['sequence']
                start, stop = int(line['start transcript coordinates'])-1, int(line['stop transcript coordinates'])-1
                start_codon, stop_codon = seq[start:start+3], seq[stop-3:stop]
                frame = int(line['frame'])
                chromosome = line['chr']
                if frame == 0 or chromosome == 'Y' or stop_codon not in ['TAA', 'TAG', 'TGA']:
                    continue
                altprots = dict()
                if trx not in trx_orfs:
                    altprots[altprot] = {'MS':int(line["MS score"]),
                                        'TE':int(line["TE score"]),
                                        'unique_pept':0,
                                        'domains':int(line["Domains"]),
                                        'start':start,
                                        'start_codon':start_codon,
                                        'stop':stop,
                                        'stop_codon':stop_codon,
                                        'chromosome':chromosome,
                                        'ORF_length':stop-start,
                                        'biotype':ensembl_trx[trx]['biotype'],
                                        'og_biotype':ensembl_trx[trx]['og_biotype'],
                                        'frame':frame,
                                        'gene_name':ensembl_trx[trx]['gene_name']
                                        }
                    trx_orfs[trx] = altprots
                else:
                    trx_orfs[trx][altprot] = {'MS':int(line["MS score"]),
                                            'TE':int(line["TE score"]),
                                            'unique_pept':0,
                                            'domains':int(line["Domains"]),
                                            'start':start,
                                            'start_codon':start_codon,
                                            'stop':stop,
                                            'stop_codon':stop_codon,
                                            'chromosome':chromosome,
                                            'ORF_length':int(line['stop transcript coordinates'])-int(line['start transcript coordinates']),
                                            'biotype':ensembl_trx[trx]['biotype'],
                                            'frame':frame,
                                            'gene_name':ensembl_trx[trx]['gene_name']
                                            }
        for trx in ensembl_trx.keys():
            if trx not in trx_orfs:
                altprots = dict()
                trx_orfs[trx] = altprots
        
        with open(self.unique_pept, 'r') as csv_:
            for line in csv_:
                ls = line.split(',')
                p_acc, tx_acc, uniq_pep = ls[1], ls[2], ls[7]
                if tx_acc in trx_orfs:
                    if p_acc in trx_orfs[tx_acc]:
                        trx_orfs[tx_acc][p_acc]['unique_pept'] = int(uniq_pep)
        return trx_orfs

    def get_rnd_trx(self, ensembl_trx, trx_orfs):
        gene_trxps = dict()
        for trx, orfs in trx_orfs.items():
            if ensembl_trx[trx]["biotype"] != "protein_coding" or not any([x.startswith("ENSP") for x in trx_orfs[trx].keys()]):
                continue
            if ensembl_trx[trx]["tsl"] != 'tsl1':
                continue
            if len(ensembl_trx[trx]["sequence"]) > 30000:
                continue
            for orf, attrs in orfs.items():
                gene = attrs["gene_name"]
                if gene not in gene_trxps:
                    gene_trxps[gene] = [trx]
                else:
                    gene_trxps[gene].append(trx)
        selected_trxps = []
        selected_genes = []
        random.seed(5)
        for gene, trxps in gene_trxps.items():
            trx = random.choice(trxps)
            selected_trxps.append(trx)
            selected_genes.append(gene)
        return selected_trxps, selected_genes

    def dataset(self, ensembl_trx, trx_orfs):
        #selected_trxps, _ = self.get_rnd_trx(ensembl_trx, trx_orfs)
        dataset = dict()
        for trx, orfs in tqdm(trx_orfs.items()):
            #if trx not in selected_trxps:
            #    continue
            if ensembl_trx[trx]['biotype'] != 'protein_coding':
                continue
            seq, seq_len = ensembl_trx[trx]['sequence'], len(ensembl_trx[trx]['sequence'])
            seq_tensor = torch.zeros(1, seq_len).view(-1)
            for orf, attrs in orfs.items():
                start, stop = attrs['start'], attrs['stop']
                if orf.startswith('ENSP'):
                    seq_tensor = map_cds(seq_tensor, start, stop, 1)
            if 1 in seq_tensor:
                dataset[trx] = {'mapped_seq': map_seq(seq),
                                'mapped_cds': seq_tensor,
                                'gene_name': ensembl_trx[trx]['gene_name']}
        return dataset

    def alt_dataset(self, ensembl_trx, trx_orfs):
        _, selected_genes = self.get_rnd_trx(ensembl_trx, trx_orfs)
        dataset = dict()
        for trx, orfs in tqdm(trx_orfs.items()):
            seq, seq_len = ensembl_trx[trx]['sequence'], len(ensembl_trx[trx]['sequence'])
            if ensembl_trx[trx]['gene_name'] in selected_genes:
                continue
            if len(find_orfs(seq)) != len([x for x in orfs.keys()]):
                continue
            if ensembl_trx[trx]['biotype'] not in ['pseudogene', 'processed_transcript']:
                continue
            seq_tensor = torch.zeros(1, seq_len).view(-1)
            for orf, attrs in orfs.items():
                start, stop = attrs['start'], attrs['stop']
                if attrs['MS'] >= 3:
                    seq_tensor = map_cds(seq_tensor, start, stop, 1)
            if 1 in seq_tensor and seq_len < 30000:
                dataset[trx] = {'mapped_seq': map_seq(seq),
                                'mapped_cds': seq_tensor,
                                'gene_name': ensembl_trx[trx]['gene_name']}
        return dataset

    def split_dataset(self, dataset, bins):
        for idx, bin_ in enumerate(bins):
            X_train = [y['mapped_seq'] for x,y in dataset.items() if x not in bin_]
            y_train = [y['mapped_cds'] for x,y in dataset.items() if x not in bin_]
            train_split = (X_train,y_train)
            pickle.dump(train_split, open(f'data/train_split{idx}.pkl', 'wb'))

            X_test = [y['mapped_seq'] for x,y in dataset.items() if x in bin_]
            y_test = [y['mapped_cds'] for x,y in dataset.items() if x in bin_]
            test_split = (X_test,y_test)
            pickle.dump(test_split, open(f'data/test_split{idx}.pkl', 'wb'))