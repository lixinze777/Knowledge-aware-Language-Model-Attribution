import pandas as pd
import json

selected_graphs = json.load(open("../rerank/questions_selected_graphs.json"))
full_graphs = list(pd.read_csv("../generated_questions/round1.csv")["knowledge_graph"])

gold_all = 1085*2
correct = 0
cite_all = 0
for i in range(len(full_graphs)):
    gold_qid1 = json.loads(full_graphs[i])[0]['qid']
    gold_qid2 = json.loads(full_graphs[i])[1]['qid']

    for graph in selected_graphs[i][1]:
        qid  = graph["qid"]
        cite_all += 1
        if qid == gold_qid1 or qid == gold_qid2:
            correct += 1

print("precision")
print(float(correct/cite_all))
print("recall")
print(float(correct/gold_all))
