import argparse
from ..modules.build_dataset import *
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('data')
args = parser.parse_args()

if __name__ == "__main__":
        ensembl_trx = pickle.load(open(args.data, 'rb'))

        dataset = dict()
        for trx, attrs in ensembl_trx.items():
            seq, seq_len, chr = attrs['sequence'], len(attrs['sequence']), attrs['chromosome']
            seq_tensor = torch.zeros(seq_len)
            dataset[trx] = {'mapped_seq': seq,
                            'mapped_cds': seq_tensor.view(1,-1),
                            'chromosome': chr}
            
        pickle.dump(dataset, open('dataset_.pkl', 'wb'))

        #alt_dataset = Data.alt_dataset(ensembl_trx)
        #Data.split_dataset(alt_dataset, 'alt_split')