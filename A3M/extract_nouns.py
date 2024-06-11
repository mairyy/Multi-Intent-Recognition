import spacy
import os
import pandas as pd
import jsonlines

def extract_nouns(text_file, nlp, dataset, nouns_dir='../datasets/MIntRec/nouns'):
    print("Beginning extracting nouns from", dataset, "...")
    noun_list = ['NNP', 'NNPS', 'NN', 'NNS']
    text = pd.read_csv(text_file, sep='\t')
    text = text['text']
    results = []

    for t in text:
        dict_ = {}
        dict_['text'] = t
        dict_['nouns'] = []
        dict_['tags'] = []

        pos_doc = [nlp(x) for x in t.split()]
        for i in range(len(pos_doc)):
            for j in range(len(pos_doc[i])):
                if pos_doc[i][j].tag_ in noun_list:
                    dict_['nouns'].append(str(pos_doc[i][j]))
                    dict_['tags'].append(str(pos_doc[i][j].tag_))

        results.append(dict_)

    with jsonlines.open(nouns_dir + '/' + dataset + '.json', 'w') as out:
        for r in results:
            out.write(r)

    print("Done! Saved at ", nouns_dir + '/' + dataset, '.json')

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    nouns_path = 'datasets/MIntRec/nouns'
    if not os.path.exists(nouns_path):
        os.makedirs(nouns_path)
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    for f in files:
        extract_nouns('datasets/MIntRec/' + f, nlp, f[:len(f)-4], nouns_path)