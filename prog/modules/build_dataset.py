import csv
import pyfaidx
import pickle
import torch
import random
from .utils import *

class Data:
    def __init__(self, OP_tsv, Ens_trx, trx_fasta):
        
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
                if "ENST" in line['transcript accession']:
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
            for n, row in enumerate(reader):
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
                    'biotype_lev2': line["Transcript type"],
                    'biotype': self.biotype_grouping[line["Transcript type"]],
                    'chromosome': line['Chromosome/scaffold name'],
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
                if not any(x in line["protein accession numbers"] for x in ["IP_", "ENSP", "II_"]):
                    continue
                trx = line["transcript accession"].split(".")[0]
                prot_id = line["protein accession numbers"].split(".")[0]
                seq = ensembl_trx[trx]['sequence']
                start, stop = int(line['start transcript coordinates'])-1, int(line['stop transcript coordinates'])-1
                start_codon, stop_codon = seq[start:start+3], seq[stop-3:stop]
                frame = int(line['frame'])
                chromosome = line['chr']
                #if start_codon not in ['ATG'] or stop_codon not in ['TAA', 'TAG', 'TGA'] or frame == 0:
                #    continue
                altprots = dict()
                if trx not in trx_orfs:
                    altprots[prot_id] = {'MS':int(line["MS score"]),
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
                                        'frame':frame,
                                        'gene_name':ensembl_trx[trx]['gene_name']}
                    trx_orfs[trx] = altprots
                else:
                    trx_orfs[trx][prot_id] = {'MS':int(line["MS score"]),
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
                                            'gene_name':ensembl_trx[trx]['gene_name']}
        for trx in ensembl_trx.keys():
            if trx not in trx_orfs:
                altprots = dict()
                trx_orfs[trx] = altprots
        return trx_orfs

    def get_candidate_list(self, trx_orfs, ensembl_trx):
        candidate_list = []
        for trx, orfs in trx_orfs.items():
            biotype = ensembl_trx[trx]['biotype']
            #exclude if not protein coding
            if biotype != 'protein_coding':
                continue
            #exclude if multiple ENSP
            if len([x for x in ensembl_trx[trx]['orf_accessions'] if x.startswith('ENSP')]) > 1:
                continue
            #exclude if len over 30000 
            if len(ensembl_trx[trx]['sequence']) > 30000:
                continue
            #exclude if noncanonical start 
            skip = False
            for orf, attrs in orfs.items():
                if orf.startswith('ENSP') and attrs['start_codon'] not in ['ATG', 'CTG', 'GTG', 'TTG']:
                    skip = True
            #exclude if noncanonical stop 
                elif orf.startswith('ENSP') and attrs['stop_codon'] not in ['TAA', 'TAG', 'TGA']:
                    skip = True
            if skip:
                continue
            #exclude if no ENSP and protein coding
            if biotype == 'protein_coding' and not any(x.startswith('ENSP') for x in orfs.keys()):
                continue

            candidate_list.append(trx)
        return set(candidate_list)


    def get_trx_list(self, ensembl_trx, trx_orfs, candidate_list):
        gene_trxps = dict()
        for trx, orfs in trx_orfs.items():
            if trx not in candidate_list:
                continue
            tsl = ensembl_trx[trx]['tsl'].split(' ')[0]
            for attrs in orfs.values():
                gene = attrs["gene_name"]
                if gene not in gene_trxps:
                    gene_trxps[gene] = [(trx, tsl)]
                else:
                    gene_trxps[gene].append((trx, tsl))
        trx_list = []
        random.seed(42)
        for gene, trxps in gene_trxps.items():
            sorted_trx = sorted(trxps, key=lambda x: x[1])
            max_tsl = sorted_trx[0][-1]
            max_tsl_trx = [trx[0] for trx in sorted_trx if trx[-1] == max_tsl]
            rnd_trx = random.choice(max_tsl_trx)
            trx_list.append(rnd_trx)
        return trx_list
    
    def dataset(self, ensembl_trx, trx_orfs, candidate_list):
        trx_list = self.get_trx_list(ensembl_trx, trx_orfs, candidate_list)
        dataset = dict()
        for trx in trx_list:
            seq, seq_len, chr = ensembl_trx[trx]['sequence'], len(ensembl_trx[trx]['sequence']), ensembl_trx[trx]['chromosome']
            seq_tensor = torch.zeros(seq_len)
            for orf, attrs in trx_orfs[trx].items():
                start, stop = attrs['start'], attrs['stop']
                if orf.startswith('ENSP'):
                    seq_tensor[start:stop] = 1

            dataset[trx] = {'mapped_seq': seq,
                            'mapped_cds': seq_tensor.view(1,-1),
                            'chromosome': chr}
        return dataset

    def alt_dataset(ensembl_trx):
        dataset = dict()
        for trx, attrs in ensembl_trx.items():
            seq, seq_len, chr = attrs['sequence'], len(attrs['sequence']), attrs['chromosome']
            seq_tensor = torch.zeros(seq_len)
            dataset[trx] = {'mapped_seq': seq,
                            'mapped_cds': seq_tensor.view(1,-1),
                            'chromosome': chr}
        return dataset

    def split_dataset(dataset, tag):
        chr_splits = [('1','7','13','19'),('2','8','14','20'),('3','9','15','21'),('4','10','16','22'),('5','11','17','X'), ('6','12','18','Y')]
        rev_chr_splits = chr_splits[::-1]
        for idx, chr_split in enumerate(chr_splits):
            X_valid = [x['mapped_seq'] for x in dataset.values() if x['chromosome'] in rev_chr_splits[idx]]
            y_valid = [x['mapped_cds'] for x in dataset.values() if x['chromosome'] in rev_chr_splits[idx]]
            X_train = [x['mapped_seq'] for x in dataset.values() if x['chromosome'] not in chr_split and x['chromosome'] not in rev_chr_splits[idx]]
            y_train = [x['mapped_cds'] for x in dataset.values() if x['chromosome'] not in chr_split and x['chromosome'] not in rev_chr_splits[idx]]
            X_test = [x['mapped_seq'] for x in dataset.values() if x['chromosome'] in chr_split]
            y_test = [x['mapped_cds'] for x in dataset.values() if x['chromosome'] in chr_split]
            split = ((X_train,y_train), (X_valid,y_valid), (X_test,y_test), [x for x,y in dataset.items() if y['chromosome'] in chr_split])
            pickle.dump(split, open(f'{tag}{idx+1}.pkl', 'wb'))
