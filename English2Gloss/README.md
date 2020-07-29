# English2Gloss NMT
Translates english sentences to American Sign Language (ASL) gloss sentences using a transformer model.

### Based on: 
1. Speech2signs NMT https://github.com/imatge-upc/speech2signs-2017-nmt
2. Attention is all you need: A Pytorch Implementation
https://github.com/jadore801120/attention-is-all-you-need-pytorch

# Requirements
- python 3.4+
- pytorch 1.3.1
- torchtext 0.4.0
- spacy 2.2.2+
- tqdm
- dill
- numpy


# Usage

## TRANSLATE USING PRETRAINED ENG2GLOSS MODEL

### 1) Download dependencies.
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2) Download the trained eng2gloss model
Download the model and vocabulary and unzip in the repo.
https://drive.google.com/file/d/15PVrfsPG3IYJh0w4nKLgdp6eMULl7-Ty/view?usp=sharing

### 3) Translate using the model
```bash
python translate.py -data_pkl eng2gloss_data.pkl -model trained.chkpt -input translate_src.txt -output prediction.txt
```
### 4) Get BLEU scores
```bash
python get_bleu_score.py -data_pkl eng2gloss_data.pkl -trg_data ASLG-PC12/ENG-ASL_Test.en -pred_data prediction.txt
```


## TRAIN FROM SCRATCH

### 1) Download dependencies.
```bash
virtualenv env
pip install -r requirements.txt
```

### 2) Preprocess the data.
```bash
python preprocess.py -lang_src en -lang_trg en -share_vocab -save_data eng2gloss_data.pkl
```

### 3) Train the model
```bash
python train.py -data_pkl eng2gloss_data.pkl -log eng2gloss -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 64 -warmup 128000 -epoch 200
```

### 4) Translate using the model
```bash
python translate.py -data_pkl eng2gloss_data.pkl -model trained.chkpt -input translate_src.txt -output prediction.txt
```
### 5) Get BLEU scores
```bash
python get_bleu_score.py -data_pkl eng2gloss_data.pkl -trg_data ASLG-PC12/ENG-ASL_Test.en -pred_data prediction.txt
```
