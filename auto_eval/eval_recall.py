import json
import csv
import pandas as pd
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


def eval_accuracy(generated_citations, selected_graph):
    num_correct = 0
    for citation in generated_citations:
        qid = citation[0]
        property = citation[1]
        value = citation[2]
        for graph in selected_graph:
            if graph["qid"] == qid and property in graph:
                if graph[property] == value:
                    num_correct = num_correct + 1
                    break
    try:
        accuracy = float(num_correct/len(generated_citations))
    except:
        accuracy = -1
    return num_correct, len(generated_citations), accuracy


def eval_recall(generated_citations, should_have_knowledge):

    num_all = len(should_have_knowledge)
    num_hit = 0
    for item in should_have_knowledge:
        qid = item[0]
        property = item[1]
        value = item[2]
        for citation in generated_citations:
            if qid == citation[0] and property == citation[1] and value == citation[2]:
                num_hit = num_hit + 1
                break
    micro_recall = float(num_hit/num_all)
    num_irre = len(generated_citations) - num_hit
    try:
        percent_irre = float(num_irre/len(generated_citations))
    except:
        percent_irre = -1
    return num_hit, num_all, micro_recall, num_irre, percent_irre


def main(folder, filename, gold):
    answers = list(pd.read_csv("../outputs/"+filename+".csv")["Answer"])
    should_have = json.load(open("gold_used_knowledge.json"))
    selected_graphs = json.load(open("../questions_selected_graphs.json"))
    '''
    We initalize the variables needed for evaluation here
    '''
    num_total_output = 0 # number of total pieces of output for evaluation
    num_total_citation = 0 # number of total citaions generated (excluding NA)
    num_total_correct = 0 # number of total citations that match the knowledge graph
    num_total_NA = 0 # number of total [NA] generated
    accuracy_sum = 0 # sum of accuracy  (for micro accuracy calculation)
    num_total_hit = 0 # number of total citations that hit the ones should generate (for macro recall calculation)
    num_total_should = 0 # number of total citations that should have (for macro recall calculation)
    recall_sum = 0 # sum of recall (for micro recall calculation)
    num_total_irre = 0 # number of total irrelevant citations
    sum_percent_irre = 0 # sum of percent_irre (for micaro irrelevant calculation)
    not_count_precision = 0 # for macro precision
    not_count_accuracy = 0 # for macro accuracy


    for i in range(len(answers)):
        answer = answers[i]
        should_have_knowledge = should_have[i][1]
        selected_graph = selected_graphs[i][1] 
        if not selected_graph: # if selected graph is [], then we skip the evaluation
            # to do: add empty values 占位
            continue
        try:
            re.findall('\[.*?\]',answer)
        except:
            continue

        num_total_output += 1
        
        generated_citations, num_NA = extract_citations(answer)
        num_total_NA += num_NA

        num_correct, num_citation, accuracy = eval_accuracy(generated_citations, selected_graph)
        num_total_correct += num_correct
        num_total_citation += num_citation
        if accuracy != -1:
            accuracy_sum += accuracy
        else:
            not_count_accuracy += 1

        num_hit, num_should, recall, num_irre, percent_irre = eval_recall(generated_citations, should_have_knowledge)
        num_total_hit += num_hit
        num_total_should += num_should
        recall_sum += recall
        num_total_irre += num_irre
        if percent_irre != -1:
            sum_percent_irre += percent_irre
        else:
            not_count_precision += 1
    average_num_NA = num_total_NA/num_total_output
    average_num_citations = num_total_citation/num_total_output
    macro_accuracy = accuracy_sum/(num_total_output - not_count_accuracy)
    micro_accuracy = num_total_correct/num_total_citation
    macro_recall = recall_sum/num_total_output
    micro_recall = num_total_hit/num_total_should
    macro_precision = 1 - sum_percent_irre/(num_total_output - not_count_precision)
    micro_precision = 1 - num_total_irre/num_total_citation

    if gold == "no_round1":
        with open("eval_results/"+folder+"/eval_"+filename+".txt", "w") as text_file:
            text_file.write("average number of [NA]: %s\n" % average_num_NA)
            text_file.write("average number of generated citations: %s\n" % average_num_citations)
            text_file.write("accuracy (micro): %s\n" % micro_accuracy)
            text_file.write("accuracy (macro): %s\n" % macro_accuracy)
            text_file.write("recall (micro): %s\n" % micro_recall)
            text_file.write("recall (macro): %s\n" % macro_recall)
        return micro_accuracy, macro_accuracy, micro_recall, macro_recall
    else:
        with open("eval_results/"+folder+"/eval_"+filename+".txt", "a") as text_file:
            text_file.write("precision (micro): %s\n" % micro_precision)
            text_file.write("precision (macro): %s\n" % macro_precision)
        return micro_precision, macro_precision


def calculate_avg_std(num_list):
    return stat.mean(num_list), stat.stdev(num_list)

def calculate_f1(p,r):
    p = float(p)
    r = float(r)
    f1 = 2*p*r/(p+r)
    return f1

if __name__ == "__main__":

    folder = ""
    filenames = [ 
        "turbo_temp_05_simple_1009.csv"
    ]

    if not os.path.exists("eval_results/"+folder):
        os.makedirs("eval_results/"+folder)
        print("The new directory is created!")

    micro_accuracy_list = []
    macro_accuracy_list = []
    micro_recall_list = []
    macro_recall_list = []
    micro_precision_list = []
    macro_precision_list = []
    for filename in tqdm(filenames):
        micro_accuracy, macro_accuracy, micro_recall, macro_recall = main(folder, filename, "no_round1")
        micro_precision, macro_precision = main(folder, filename, "with_round1")
        micro_accuracy_list.append(micro_accuracy)
        macro_accuracy_list.append(macro_accuracy)
        micro_recall_list.append(micro_recall)
        macro_recall_list.append(macro_recall)
        micro_precision_list.append(micro_precision)
        macro_precision_list.append(macro_precision)

    micro_accuracy_mean, micro_accuracy_std = calculate_avg_std(micro_accuracy_list)
    macro_accuracy_mean, macro_accuracy_std = calculate_avg_std(macro_accuracy_list)
    micro_recall_mean, micro_recall_std = calculate_avg_std(micro_recall_list)
    macro_recall_mean, macro_recall_std = calculate_avg_std(macro_recall_list)
    micro_precision_mean, micro_precision_std = calculate_avg_std(micro_precision_list)
    macro_precision_mean, macro_precision_std = calculate_avg_std(macro_precision_list)

    micro_f1 = calculate_f1(micro_precision_mean, micro_recall_mean)
    macro_f1 = calculate_f1(macro_precision_mean, macro_recall_mean)

    with open("eval_results/"+folder+"/full_result.txt", "w") as text_file:
        text_file.write("accuracy (micro): %s   std: %s\n" % (micro_accuracy_mean, micro_accuracy_std))
        text_file.write("accuracy (macro): %s   std: %s\n" % (macro_accuracy_mean, macro_accuracy_std))
        text_file.write("recall (micro): %s   std: %s\n" % (micro_recall_mean, micro_recall_std))
        text_file.write("recall (macro): %s   std: %s\n" % (macro_recall_mean, macro_recall_std))
        text_file.write("precision (micro): %s   std: %s\n" % (micro_precision_mean, micro_precision_std))
        text_file.write("precision (macro): %s   std: %s\n" % (macro_precision_mean, macro_precision_std))
        text_file.write("f1 score (micro): %s\n" % micro_f1)
        text_file.write("f1 score (macro): %s\n" % macro_f1)