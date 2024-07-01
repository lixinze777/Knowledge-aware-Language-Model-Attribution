import spacy
from tqdm import tqdm
import json


def preprocess(nlp, sent):
    doc = nlp(sent)
    entity = [X.text for X in doc.ents]
    for i in range(len(entity)):
        entity[i] = entity[i].replace("\\","")
        if entity[i][-2:] == "'s":
            entity[i] = entity[i][:-2]
        if entity[i][-1:] == "'":
            entity[i] = entity[i][:-1]

    return list(set(entity))


if __name__ == "__main__":

    with open('../data/questions.json', 'r') as f:
        questions = json.load(f)

    nlp = spacy.load("en_core_web_sm")
    #f = open("../data/example_questions.txt", "r")
    #questions = f.readlines()

    output = []
    for question in tqdm(questions):
        entity = preprocess(nlp, question[0])
        output.append((question, entity))

    with open('retrieved_entities.json', 'w') as outfile:
        json.dump(output, outfile)
    
        