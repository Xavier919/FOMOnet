import argparse
from ..modules.build_dataset import *
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('ensembl_trx')
args = parser.parse_args()

if __name__ == "__main__":
        ensembl_trx = pickle.load(open(args.ensembl_trx, 'rb'))
        alt_dataset = data.alt_dataset(ensembl_trx)
        data.split_dataset(alt_dataset, 'alt_split')