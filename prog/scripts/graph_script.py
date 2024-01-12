#imports
import pickle
import argparse
from modules.evaluate import PR_curve, ROC_curve

parser = argparse.ArgumentParser()
parser.add_argument('split1')
parser.add_argument('split2')
parser.add_argument('split3')
parser.add_argument('split4')
parser.add_argument('split5')
parser.add_argument('split6')

args = parser.parse_args()

if __name__ == "__main__":
    split1 = pickle.load(open(args.split1, 'rb'))
    split2 = pickle.load(open(args.split2, 'rb'))
    split3 = pickle.load(open(args.split3, 'rb'))
    split4 = pickle.load(open(args.split4, 'rb'))
    split5 = pickle.load(open(args.split5, 'rb'))
    split6 = pickle.load(open(args.split6, 'rb'))

    preds = [split1[0], split2[0], split3[0], split4[0], split5[0], split6[0]]
    targets = [split1[1], split2[1], split3[1], split4[1], split5[1], split6[1]]

    PR_curve(preds, targets)
    ROC_curve(preds, targets)