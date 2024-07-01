import json
import csv
import pandas as pd
from tqdm import tqdm
import re

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


def eval_precision(all_pre_hyp_pairs):
    
    num_all_citations = 0
    num_all_correct_citations = 0
    num_all_data = 0
    sum_precision = 0
    results = []
    for pre_hyp_pairs in tqdm(all_pre_hyp_pairs):
        num_all_data += 1
        num_data_citations = 0
        num_data_correct_citations = 0
        for pre_hyp_pair in pre_hyp_pairs:
            num_all_citations += 1
            num_data_citations += 1
            text = pre_hyp_pair[0]
            value = pre_hyp_pair[1].split(': ',1)[1]

            if value.lower() in text.lower():
                num_all_correct_citations += 1
                num_data_correct_citations += 1
                results.append([text, pre_hyp_pair[1], "YES"])
            elif check_entail(text, pre_hyp_pair[1]):
                num_all_correct_citations += 1
                num_data_correct_citations += 1
                results.append([text, pre_hyp_pair[1], "YES"])
            else:
                results.append([text, pre_hyp_pair[1], "NO"])

        try:
            sum_precision += num_data_correct_citations/num_data_citations
        except:
            num_all_data = num_all_data - 1

    micro_precision = float(num_all_correct_citations/num_all_citations)
    macro_precision = float(sum_precision/num_all_data)
    return macro_precision, micro_precision, results


def extract_citations(sent):
    generated_citations = [] #  list of triple of [qid, property, value]

    citations = re.findall('\[.*?\]',sent)
    plain_text = re.sub("[\[].*?[\]]", "", sent).replace("  "," ").replace(" .", ".")

    for citation in citations:
        if citation != "[NA]":
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
                        generated_citations.append(property+": "+value)
                else:
                    value = items[i].rsplit(", ", 1)[0]
                    if property != "":
                        generated_citations.append(property+": "+value)
                    try:
                        property = items[i].rsplit(", ", 1)[1]
                    except:
                        property = ""

    return plain_text, generated_citations


def main(filename):
    all_pre_hyp_pairs = []
    answers = list(pd.read_csv("../outputs/"+filename+".csv")["Answer"])
    selected_graphs = json.load(open("../rerank/questions_selected_graphs.json"))

    for i in range(len(answers[:400])):
        selected_graph = selected_graphs[i][1] 
        answer = answers[i]

        if not selected_graph: # if selected graph is [], then we skip the evaluation
            # to do: add empty values 占位
            continue
        try:
            re.findall('\[.*?\]',answer)
        except:
            continue
        
        pre_hyp_pair = []
        sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', answer)
        for sentence in sentences:
            plain_text, generated_citations = extract_citations(sentence)

            if generated_citations:
                for c in generated_citations:
                    pre_hyp_pair.append([plain_text, c])
        all_pre_hyp_pairs.append(pre_hyp_pair)

    macro_precision, micro_precision, results = eval_precision(all_pre_hyp_pairs)
    with open("eval_results/precision_"+filename+".txt", "w") as text_file:
        text_file.write("relevance (micro): %s\n" % micro_precision)
        text_file.write("relevance (macro): %s\n" % macro_precision)

    with open("eval_results/precision_"+filename+".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Sentence","Citation","Entailment"])
        for row in results:
            writer.writerow(row)    


if __name__ == "__main__":
    filenames = [
        "llama_13b_0613",
    ]

    for filename in filenames:
        main(filename)