import argparse
import dill as pickle

from torchtext.data.metrics import bleu_score
from torchtext.data import Dataset, Example


def main():
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with vocabulary.')
    parser.add_argument('-trg_data', default='PSLG-PC12/ENG-ASL_Test.en')
    parser.add_argument('-pred_data', default='predictions.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    opt = parser.parse_args()

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']

    fields = [('src', SRC)]

    with open(opt.trg_data, 'r') as f:
        trg_loader = Dataset(examples=[Example.fromlist([x], fields) for x in f]
                            , fields={'src': SRC})
    trg_txt = [x.src for x in trg_loader]

    with open(opt.pred_data, 'r') as f:
        pred_loader = Dataset(examples=[Example.fromlist([x], fields) for x in f]
                            , fields={'src': SRC})
    pred_txt = [[x.src] for x in pred_loader]

    score = bleu_score(trg_txt, pred_txt)
    print('Bleu 4 score is {}'.format(str(score)))

    with open('bleu_score.txt', 'w') as f:
        f.write('Bleu 4 score is {}'.format(str(score)))

if __name__ == "__main__":
    '''
    Usage: python get_bleu_score.py -data_pkl eng2gloss_data.pkl -trg_data ASLG-PC12/ENG-ASL_Test.en -pred_data predictions.txt
    '''
    main()


