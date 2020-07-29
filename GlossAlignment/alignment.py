import csv 
from collections import defaultdict

class Alignment:
    def __init__(self, filename="gloss_alignment.csv"):
        self.alignment_dict = self.csv_to_dict(filename)
     
    def csv_to_dict(self, filename): 
        alignment_dict = defaultdict(list)
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader: 
                for (k, v) in row.items():
                    alignment_dict[k].append(v)
        return alignment_dict
    
    def find_match(self, word): 
        try:
            return self.alignment_dict['Pose2Idx'][self.alignment_dict['Eng2GlossVocab'].index(word)] 
        except:
            return
                                        

if __name__=="__main__":
    import torch 

    GLOSS_IN_FP = "/data2/t-arnair/roshan/Speech-to-ASL/English2Gloss/prediction.txt"
    GLOSS_OUT_FP = "/data2/t-arnair/roshan/Speech-to-ASL/English2Gloss/glosses_for_poses.txt"
    LIZ_DATA = "/data2/t-arnair/roshan/GlossAlignment/Liz.data"

    d = torch.load(LIZ_DATA)

    liz_data = torch.load(LIZ_DATA)['filename']

    # aligner = Alignment()
    aligner = Alignment("gloss_alignment_lemmatized.csv")

    g = open(GLOSS_OUT_FP, 'w')

    with open(GLOSS_IN_FP, 'r') as f:
        for line in f:
            tokens = line.replace('.', '').split()
            print('\n')
            print(line)
            print(tokens)
            out = []
            for t in tokens:
                s = aligner.find_match(t.upper())
                print(s)
                if s and s != "<UNK>":
                    r = liz_data[int(s)]
                    r = r.split('.')[0][4:]
                    out.append(r)

                    print(t.upper())
                    print(s)
                    print(r)
                    print('\n')

            g.write(','.join(out) + '\n')
    
    g.close()

