import json
import csv
from tqdm import tqdm
import re
import statistics as stat
import os

def extract_citations(answer):
    generated_citations = [] #  list of triple of [qid, property, value]
    num_NA = 0

    citations = re.findall('\[.*?\]',answer)

    for citation in citations:
        if citation == "[NA]":
            num_NA = num_NA + 1
        else:
            qid = citation.split(',')[0][1:]
            items = citation.split(": ")
            property = ""
            value = ""
            for i in range(len(items)):
                if i == 0:
                    try:
                        property = items[i].rsplit(", ", 1)[1]
                    except:
                        property = ""
                elif i == len(items)-1:
                    value = items[i][:-1]
                    if property != "":
                        generated_citations.append([qid, property, value])
                else:
                    value = items[i].rsplit(", ", 1)[0]
                    if property != "":
                        generated_citations.append([qid, property, value])
                    try:
                        property = items[i].rsplit(", ", 1)[1]
                    except:
                        property = ""

    return generated_citations, num_NA



if __name__ == "__main__":

    data = json.load(open("../round5_extended_paragraph.json"))

    golds = []

    for d in data:
        generated_citations, num_NA = extract_citations(d[0])
        golds.append(generated_citations)

    with open('gold_used_knowledge.json', 'w') as f:
        json.dump(golds, f)    

    