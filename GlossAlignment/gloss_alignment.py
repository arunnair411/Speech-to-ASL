#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import spacy 
import os
import numpy as np
import pickle
import pandas
from pandas import DataFrame

nlp = spacy.load('en_core_web_lg')
metadata = torch.load('Liz.data')
poses = metadata['gloss'] # ignore gloss_variant


# In[15]:


"""
Generate a list of the unique gloss tokens in each vocabulary 
"""
### GLOSS -> POSE VOCAB
pose2idx = {} 
for idx, pose in enumerate(poses): 
    if pose not in pose2idx: 
        pose2idx[str(pose)] = idx 

gloss2posevocab = list(pose2idx.keys())
gloss2posevocab.sort()


# In[16]:


### ENG -> GLOSS VOCAB
eng2glossvocab = [] 
with open('sample-corpus-asl-en.asl', encoding="utf8") as f:
    for line in f:
        for word in line.split(" "): 
            word = word.strip()
            if word not in eng2glossvocab: 
                eng2glossvocab.append(word)

eng2glossvocab.sort()
print("Gloss2Pose vocab: {} tokens, Eng2Gloss vocab: {} tokens".format(len(gloss2posevocab), len(eng2glossvocab)))


# In[59]:


"""
Preprocess the gloss vocab from gloss2pose and map the simplified gloss tokens to the original gloss tokens 
"""
import re 
import string

def strip(text): 
    strip_again = False 
    if "+" in text:
        text = text[:text.index("+")]
        strip_again = True 
    if ")" in text: 
        text = text[text.index(")")+1:]
        strip_again = True 
    if "_" in text: 
        text = text[:text.index("_")]
        strip_again = True 
    return strip_again, text 
    
def preprocess(word): 
    candidates = []
    token = nlp(word) 
    num_tokens = len(token)
    dash_concat = False 
    for idx, tok in enumerate(token): 
        if not tok.is_punct: 
            strip_again, new_tok = strip(tok.text)
            while strip_again: 
                strip_again, new_tok = strip(new_tok)
            candidates.append(new_tok)
            
            # concatenate consecutive tokens linked by a dash 
            if dash_concat: 
                candidates.append(concat_tok+new_tok)
                dash_concat = False 
            if idx != num_tokens - 1 and token[idx+1].text == "-":
                dash_concat = True 
                concat_tok = new_tok
            
    return candidates 
    
def build_simplified_gloss(vocab, dict_out): 
    for word in vocab: 
        simplified = preprocess(word)
        # if the gloss token is standalone, give it priority as a key in the mapping  
        if len(simplified) == 1: 
            dict_out[simplified[0]] = word 
        else: 
            for s in simplified: 
                if s not in dict_out: 
                    dict_out[s] = word 


# In[61]:


""" EXAMPLE """
sample_vocab = ["POP++AGENT", "PING-PONG/TENNIS", "OPPOSITE+AGENT", "BOX_2", "DAY_2/BIRTHDAY_2", "CUTE++DCL'cane'", "(2h)#BACK_2",
          "ONE-FOURTH", "(1h)", "LIKE+(1h)NEG", "ALL-NIGHT-AFTER-MIDNIGHT", "ICL'peeling an orange'", "(L)LEATHER", "FIVE_2+HOUR"]
sample_dict = {}

build_simplified_gloss(sample_vocab, sample_dict)
print(sample_dict)


# In[62]:


""" Build gloss dictionary for Gloss2Pose Vocab """

simplifiedgloss2pose = {}
build_simplified_gloss(gloss2posevocab, simplifiedgloss2pose)


# In[64]:


"""
eng2glossvocab -> simplified eng2glossvocab -> simplifiedgloss2pose -> gloss2pose -> idx for pose 
"""
simplified_eng2gloss_col = []
simplified_gloss2pose_col = []
pose2idx_col = [] # Note: the index is needed to look up the corresponding pose

for gloss in eng2glossvocab: 
    # Simple stripping of prefix / annotations
    if "-" in gloss: 
        gloss = gloss[gloss.index("-")+1:]
    simplified_eng2gloss_col.append(gloss)
    if gloss in simplifiedgloss2pose: 
        simplified_gloss2pose_col.append(simplifiedgloss2pose[gloss])
        pose2idx_col.append(pose2idx[simplifiedgloss2pose[gloss]])
    else: 
        simplified_gloss2pose_col.append("<UNK>")
        pose2idx_col.append("<UNK>")


# In[69]:


data = {'Eng2GlossVocab': eng2glossvocab,
        'Simplified Eng2Gloss': simplified_eng2gloss_col,
        'Gloss2Pose': simplified_gloss2pose_col, 
        'Pose2Idx': pose2idx_col
        }

df = DataFrame(data)
df.to_csv("gloss_alignment.csv")
print(df)

