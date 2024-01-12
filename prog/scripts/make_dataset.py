import argparse
from ..modules.build_dataset import *
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('data')
args = parser.parse_args()

if __name__ == "__main__":
        ensembl_trx = pickle.load(open(args.data, 'rb'))

        alt_dataset = Data.alt_dataset(ensembl_trx)
        Data.split_dataset(alt_dataset, 'alt_split')