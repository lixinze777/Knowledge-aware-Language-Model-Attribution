import json
import csv
import pandas as pd
import re
from tqdm import tqdm


def main():
    questions = json.load(open("../data/questions.json"))

    all_extracted_knowledge = []
 
    # process round 1 knowledge
    df = pd.read_csv("../generated_questions/round1.csv")
    kgs = list(df["knowledge_graph"])
    answers = list(df["answer"])
    for i in range(len(answers)):
        knowledge = re.findall('\[.*?\]',answers[i])
        extracted_knowledge = []
        actual_kg1 = json.loads(kgs[i])[0]
        actual_kg2 = json.loads(kgs[i])[1]
        for k in knowledge:
            try:
                qid = k.split(", ")[0].split(": ")[1].replace('[','').replace(']','')
                property = k.split(", ")[1].split(": ")[0].replace('[','').replace(']','')
                value = k.split(", ")[1].split(": ")[1].replace('[','').replace(']','')

                if qid == actual_kg1["qid"] and value == actual_kg1[property]:
                    if [qid, property, value] not in extracted_knowledge:
                        extracted_knowledge.append([qid, property, value])
                elif qid == actual_kg2["qid"] and value == actual_kg2[property]:
                    if [qid, property, value] not in extracted_knowledge:
                        extracted_knowledge.append([qid, property, value])
                else:
                    pass
            except:
                pass
        all_extracted_knowledge.append(extracted_knowledge)
    '''
    for i in range(1085):
        all_extracted_knowledge.append([])
    '''
    for round in tqdm(range(2,7)): # round 2 to round 6

        df = pd.read_csv('../generated_questions/round'+str(round)+'.csv')
        knowledge = list(df["knowledge"])

        for i in range(len(knowledge)):
            stop_round = questions[i][1]
            if stop_round >= round:
                qid = knowledge[i].split("qid: ")[1].split(",")[0]
                property = knowledge[i].rsplit(", ",1)[1].split(":")[0]
                try:
                    value = knowledge[i].rsplit(": ",1)[1].replace('[','').replace(']','')
                    if value[-1] == '"':
                        value = value[:-1]
                except:
                    print(knowledge[i])
                if "Category" not in value:
                    if [qid, property, value] not in all_extracted_knowledge[i]:
                        all_extracted_knowledge[i].append([qid, property, value])

    output = []
    lens = []
    for i in range(len(questions)):
        output.append([questions[i][0],all_extracted_knowledge[i]])
        lens.append(len(all_extracted_knowledge[i]))

    with open('gold_used_knowledge_with_round1.json', 'w') as f:
        json.dump(output, f)
    with open('num_used_knowledge_with_round1.json', 'w') as f:
        json.dump(lens, f)

if __name__ == "__main__":
    main()
