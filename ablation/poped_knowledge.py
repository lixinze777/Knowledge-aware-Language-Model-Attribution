import logging
logger = logging.getLogger(__name__)
import argparse
import json
import pandas as pd


def main(num_remove):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type = str, default = "../prompt/generate_prompt_sentlevel.json", help = "Path to the prompt file")
    parser.add_argument("--input_path", type = str, default = "../rerank/questions_selected_graphs.json", help = "Path to the input file")
    parser.add_argument("--output_path", type = str, default = "../outputs/ablation_NA_"+str(num_remove)+".csv", help = "Path to the output file")
    parser.add_argument("--temperature", type=float, default=0.5, help= "Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help= "Nucleus sampling top-p")
    parser.add_argument("--model", type=str, default = "gpt-3.5-turbo-0301", help= "Model to use")
    parser.add_argument("--rerun", type=int, default = 3, help = "number of times to re-run the query when reach openai api rate limit")
    args = parser.parse_args()

    questions = json.load(open(args.input_path))
    gt_kg = list(pd.read_csv("../generated_questions/round1.csv")["knowledge_graph"])
    should_have = json.load(open("../auto_eval/gold_used_knowledge_no_round1.json"))

    poped_kg = []

    for i in range(len(gt_kg)):
        _kg = json.loads(gt_kg[i])
        _know = should_have[i][1]

        tups = list(_kg[0].items())
        _kg[0] = dict(tups[-1:]+tups[:-1])
        tups = list(_kg[1].items())
        _kg[1] = dict(tups[-1:]+tups[:-1])

        for j in range(num_remove):
            if j >= len(_know):
                break
            qid = _know[j][0]
            property = _know[j][1].replace(']\"','')
            try:
                if _kg[0]["qid"] == qid:
                    poped_kg.append(property+": "+_kg[0][property])
                elif _kg[1]["qid"] == qid:
                    poped_kg.append(property+": "+_kg[1][property])
            except:
                pass

    with open('poped'+str(num_remove)+'.json', 'w') as f:
        json.dump(poped_kg, f)

if __name__ == "__main__":
    main(1)
    main(2)
    main(3)