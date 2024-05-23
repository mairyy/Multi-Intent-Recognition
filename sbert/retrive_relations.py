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

def get_relation(relation, knowledge, heads):
    relation_outputs = []

    for h in heads:
        rows = knowledge[knowledge[0] == h]
        relation_outputs.append(rows[rows[1] == relation][2].unique()[0])

    return relation_outputs

if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")

    relation_types = ["xReact", "xWant"]
    atomic20 = pd.read_csv('datasets/atomic20.tsv', sep='\t', header=None)
    
    input_paths = ['inputs/train.tsv', 'inputs/dev.tsv', 'inputs/test.tsv']
    output_paths = ['outputs/sbert_train.csv', 'outputs/sbert_dev.csv', 'outputs/sbert_test.csv']

    for i, path in tqdm(enumerate(input_paths), total=len(input_paths)):
        input = pd.read_csv(path, sep='\t')
        sentences = list(input['text'])
        outputs = pd.DataFrame(sentences, columns=['text'])
        for relation in relation_types:
            relation_knowledge = atomic20[atomic20[1] == relation]
            relation_knowledge = set(relation_knowledge[0])
            relation_knowledge = list(relation_knowledge)

            heads = get_sim_head(model=model, knowledge=relation_knowledge, sentences=sentences)
            outputs[relation] = get_relation(relation, atomic20, heads)

        outputs.to_csv(output_paths[i])
        print("Done", path)

