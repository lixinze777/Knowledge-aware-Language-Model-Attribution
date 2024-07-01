import json
import csv
import pandas as pd
from tqdm import tqdm
import re
import os

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

global autoais_model, autoais_tokenizer
AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"
autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16)
autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)


def check_entail(premise, hypothesis):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(premise, hypothesis)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def extract_citations(answer):
    generated_citations = [] #  list of triple of [qid, property, value]
    citations = re.findall('\[.*?\]',answer)

    for citation in citations:
        if citation != "[NA]":
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

    return generated_citations


def get_dont_have_knowledge(generated_citations, should_have_knowledge):

    dont_have_knowledge = []
 
    for item in should_have_knowledge:
        qid = item[0]
        property = item[1]
        value = item[2]
        hit = False
        for citation in generated_citations:
            if qid == citation[0] and property == citation[1] and value == citation[2]:
                hit = True
                break
        if not hit:
            dont_have_knowledge.append(property+": "+value)
    return dont_have_knowledge


def get_NA_sentences(sentences):
    NAsentences = []
    for sentence in sentences:
        if "[NA]" in sentence:
            NAsentences.append(re.sub("[\[].*?[\]]", "", sentence).replace("  "," ").replace(" .", "."))
    return NAsentences


def eval_NA(NA_sentences, dont_have_knowledge):
    
    if not NA_sentences:
        return -1, [], -1, -1
    
    if not dont_have_knowledge:
        return 0, [], len(NA_sentences), len(dont_have_knowledge)
    
    matched = 0
    matched_result = []
    for sent in NA_sentences:
        for knowledge in dont_have_knowledge:
            if check_entail(sent, knowledge):
                matched += 1
                matched_result.append([sent, knowledge])
    return matched, matched_result, len(NA_sentences), len(dont_have_knowledge)



def main(folder, filename, gold):
    answers = list(pd.read_csv("../outputs/"+filename+".csv")["Answer"])
    should_have = json.load(open("../auto_eval/gold_used_knowledge_"+gold+".json"))
    #selected_graphs = json.load(open("../rerank/questions_selected_graphs.json"))
    '''
    We initalize the variables needed for evaluation here
    '''
    all_matched = []
    num_matched = 0
    num_all_na = 0
    num_all_know = 0
    for i in tqdm(range(len(answers))):
        answer = answers[i]
        should_have_knowledge = should_have[i][1]
        '''
        selected_graph = selected_graphs[i][1] 
        if not selected_graph: # if selected graph is [], then we skip the evaluation
            # to do: add empty values 占位
            continue
        '''
        try:
            re.findall('\[.*?\]',answer)
        except:
            continue
        
        generated_citations = extract_citations(answer)
        dont_have_knowledge = get_dont_have_knowledge(generated_citations, should_have_knowledge) # hypothesis

        sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', answer)
        NA_sentences = get_NA_sentences(sentences)  # PREMISE

        matched, matched_result, num_NA_sentences, num_dont_have_knowledge = eval_NA(NA_sentences, dont_have_knowledge)
        if matched != -1:
            num_matched += matched
            num_all_na += num_NA_sentences
            num_all_know += num_dont_have_knowledge
            if matched > 0:
                all_matched.append(matched_result)

    precision = float(num_matched/num_all_na)
    recall = float(num_matched/num_all_know)
    
    return precision, recall, all_matched


if __name__ == "__main__":

    folder = "chatgpt_temp05"
    filenames = [ 
        "ablation_NA_3",
    ]

    if not os.path.exists("eval_results/"+folder):
        os.makedirs("eval_results/"+folder)
        print("The new directory is created!")

    for filename in filenames:
        precision, recall, all_matched = main(folder, filename, "no_round1")
        with open("eval_results/"+folder+"/"+filename+"_NA.txt", "w") as text_file:
            text_file.write("precision (micro): %s\n" % precision)
            text_file.write("recall (micro): %s\n" % recall)   
        with open("eval_results/"+folder+"/"+filename+"_matched.json", 'w') as outfile:
            json.dump(all_matched, outfile)
