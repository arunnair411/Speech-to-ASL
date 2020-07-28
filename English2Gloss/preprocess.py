''' Handling the data io '''
import os
import argparse
import logging
import dill as pickle
import urllib
from tqdm import tqdm
import sys
import codecs
import spacy
import torch
import tarfile
import torchtext.data
import torchtext.datasets
from torchtext.datasets import TranslationDataset
import transformer.Constants as Constants


__author__ = "Yu-Hsiang Huang"


def main():
    '''
    Usage: python preprocess.py -lang_src de -lang_trg en -save_data multi30k_de_en.pkl -share_vocab
    '''

    spacy_support_langs = ['de', 'el', 'en', 'es', 'fr', 'it', 'lt', 'nb', 'nl', 'pt']

    parser = argparse.ArgumentParser()
    parser.add_argument('-lang_src', required=True, choices=spacy_support_langs)
    parser.add_argument('-lang_trg', required=True, choices=spacy_support_langs)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-data_src', type=str, default=None)
    parser.add_argument('-data_trg', type=str, default=None)

    parser.add_argument('-max_len', type=int, default=100)
    parser.add_argument('-min_word_count', type=int, default=3)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    #parser.add_argument('-ratio', '--train_valid_test_ratio', type=int, nargs=3, metavar=(8,1,1))
    #parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    assert not any([opt.data_src, opt.data_trg]), 'Custom data input is not support now.'
    assert not any([opt.data_src, opt.data_trg]) or all([opt.data_src, opt.data_trg])
    print(opt)

    src_lang_model = spacy.load(opt.lang_src)
    trg_lang_model = spacy.load(opt.lang_trg)

    STOP_WORDS = ['X-', 'DESC-']

    def tokenize_src(text):
        for w in STOP_WORDS:
            text = text.replace(w, '')
        return [tok.text for tok in src_lang_model.tokenizer(text)]

    def tokenize_trg(text):
        for w in STOP_WORDS:
            text = text.replace(w, '')
        return [tok.text for tok in trg_lang_model.tokenizer(text)]

    SRC = torchtext.data.Field(
        tokenize=tokenize_src, lower=not opt.keep_case,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)

    TRG = torchtext.data.Field(
        tokenize=tokenize_trg, lower=not opt.keep_case,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)

    MAX_LEN = opt.max_len
    MIN_FREQ = opt.min_word_count

    if not all([opt.data_src, opt.data_trg]):
        assert {opt.lang_src, opt.lang_trg} == {'en', 'en'}
    else:
        # Pack custom txt file into example datasets
        raise NotImplementedError

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    TRAIN_SRC_FN = 'ASLG-PC12/ENG-ASL_Train.en'
    TRAIN_TRG_FN = 'ASLG-PC12/ENG-ASL_Train.asl'
    VAL_SRC_FN = 'ASLG-PC12/ENG-ASL_Dev.en' 
    VAL_TRG_FN = 'ASLG-PC12/ENG-ASL_Dev.asl'
    TEST_SRC_FN = 'ASLG-PC12/ENG-ASL_Test.en'
    TEST_TRG_FN = 'ASLG-PC12/ENG-ASL_Test.asl'

    with open(TRAIN_SRC_FN, 'r') as f:
        train_src = list(f)

    with open(TRAIN_TRG_FN, 'r') as f:
        train_trg = list(f)

    with open(VAL_SRC_FN, 'r') as f:
        val_src = list(f)

    with open(VAL_TRG_FN, 'r') as f:
        val_trg = list(f)

    with open(TEST_SRC_FN, 'r') as f:
        test_src = list(f)

    with open(TEST_TRG_FN, 'r') as f:
        test_trg = list(f)

    fields = [('src', SRC), ('trg', TRG)]

    train = torchtext.data.Dataset(
                    examples=[torchtext.data.Example.fromlist(x, fields) for x in zip(train_src, train_trg)]
                    , fields=fields
                    , filter_pred=filter_examples_with_length)

    val = torchtext.data.Dataset(
                    examples=[torchtext.data.Example.fromlist(x, fields) for x in zip(val_src, val_trg)]
                    , fields=fields
                    , filter_pred=filter_examples_with_length)
    
    test = torchtext.data.Dataset(
                    examples=[torchtext.data.Example.fromlist(x, fields) for x in zip(test_src, test_trg)]
                    , fields=fields
                    , filter_pred=filter_examples_with_length)

    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    print('[Info] Get source language vocabulary size:', len(SRC.vocab))
    TRG.build_vocab(train.trg, min_freq=MIN_FREQ)
    print('[Info] Get target language vocabulary size:', len(TRG.vocab))

    if opt.share_vocab:
        print('[Info] Merging two vocabulary ...')
        for w, _ in SRC.vocab.stoi.items():
            # TODO: Also update the `freq`, although it is not likely to be used.
            if w not in TRG.vocab.stoi:
                TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
        TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
        for w, i in TRG.vocab.stoi.items():
            TRG.vocab.itos[i] = w
        SRC.vocab.stoi = TRG.vocab.stoi
        SRC.vocab.itos = TRG.vocab.itos
        print('[Info] Get merged vocabulary size:', len(TRG.vocab))


    data = {
        'settings': opt,
        'vocab': {'src': SRC, 'trg': TRG},
        'train': train.examples,
        'valid': val.examples,
        'test': test.examples}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))


if __name__ == '__main__':
    main()
