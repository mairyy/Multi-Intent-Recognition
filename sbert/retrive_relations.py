from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from tqdm import tqdm

def get_sim_head(model, knowledge, sentences):
    # Compute embedding
    knowledge_embedding = model.encode(knowledge, convert_to_tensor=True)
    sentences_embedding = model.encode(sentences, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(sentences_embedding, knowledge_embedding)
    rel_index = torch.argmax(cosine_scores, dim=1)

    heads = []
    for i in range(len(rel_index)):
        heads.append(knowledge[rel_index[i]])
        
    return heads

def get_relation(relation_types, knowledge, heads, sentences):
    df = pd.DataFrame(sentences, columns=["text"])
    relations = {}
    for r in relation_types:
        relations[r] = []

    for h in heads:
        rows = knowledge[knowledge[0] == h]
        rel_list = rows[1].unique()
        for r in relation_types:
            if r in rel_list:
                relations[r].append(rows[rows[1] == r][2].unique()[0])
            else:
                relations[r].append('none')

    for r in relation_types:
        df[r] = relations[r]

    return df

if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")

    relation_types = ["xReact", "xWant"]
    atomic20 = pd.read_csv('datasets/atomic20.tsv', sep='\t', header=None)
    atomic20 = atomic20[atomic20[1].isin(relation_types)]

    knowledge = set(atomic20[0])
    knowledge = list(knowledge)
    
    input_paths = ['inputs/train.tsv', 'inputs/dev.tsv', 'inputs/test.tsv']
    outpu_paths = ['outputs/sbert_train.csv', 'outputs/sbert_dev.csv', 'outputs/sbert_test.csv']

    for i, path in tqdm(enumerate(input_paths)):
        input = pd.read_csv(path, sep='\t')
        sentences = list(input['text'])
        heads = get_sim_head(model=model, knowledge=knowledge, sentences=sentences)
        outputs = get_relation(relation_types, atomic20, heads, sentences)
        outputs.to_csv(outpu_paths[i])
        print("Done", path)

