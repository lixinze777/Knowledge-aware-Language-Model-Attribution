import logging
logger = logging.getLogger(__name__)
import argparse
import openai
from tqdm import tqdm
import json
import asyncio
import csv
from typing import Any
import pandas as pd
import random

openai.api_key = "Your Key"

class ChatGPT:

    def __init__(self, args):
        self.args = args

    async def dispatch_openai_requests(
        self,
        message_list: list[list[dict[str,Any]]],
        model: str,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Dispatches requests to OpenAI API asynchronously.
        
        Args:
            messages_list: List of messages to be sent to OpenAI Completion API.
            model: OpenAI model to use.
            temperature: Temperature to use for the model.
            max_tokens: Maximum number of tokens to generate.
            top_p: Top p to use for the model.
        Returns:
            List of responses from OpenAI API.
        """
        async_responses = [
            openai.ChatCompletion.acreate(
                model=self.args.model,
                messages=x,
                temperature=temperature,
                top_p=top_p,
            )
            for x in message_list
        ]
        return await asyncio.gather(*async_responses)


    def generate(self, message_list):
        predictions = asyncio.run(
                self.dispatch_openai_requests(
                    message_list=message_list,
                    model=self.args.model,
                    temperature=self.args.temperature,
                    top_p = self.args.top_p
                    )
        )
        
        outputs = []
        for i, x in enumerate(predictions):
            output = x['choices'][0]['message']['content']
            outputs.append(output)

        return outputs


    def save_output(self, question_list, kg_list, outputs):
        rows = zip(question_list, kg_list, outputs)
        with open(self.args.output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Question","Knowledge Graph","Answer"])
            for row in rows:
                writer.writerow(row)    


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

    chatgpt = ChatGPT(args)
    questions = json.load(open(args.input_path))
    example = json.load(open(args.prompt_path))
    gt_kg = list(pd.read_csv("../generated_questions/round1.csv")["knowledge_graph"])
    should_have = json.load(open("../auto_eval/gold_used_knowledge_no_round1.json"))

    replaced_kg = []

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
                    _kg[0].pop(property)
                elif _kg[1]["qid"] == qid:
                    _kg[1].pop(property)
            except:
                pass
        replaced_kg.append([_kg[0],_kg[1]])

    for i in range(len(questions)):
        questions[i][1] = json.dumps(replaced_kg[i])

    eg_question = example[0]
    eg_answer = example[1]
    eg_kg = str(example[2]).replace("{'","{").replace("': '",": ").replace("', '",", ").replace("'}","}")
    prefix = "Considering the information: "
    prompt_input = prefix + eg_kg + ", " + eg_question

    outputs = []
    question_list = []
    kg_list = []
    assert(len(questions)==1085) # check data length
    packs = [questions[x:x+5] for x in range(0, len(questions), 5)] # pack questions to packs of 5

    logger.info("----Initializing Generation----")
    for pack in tqdm(packs):
        message_list = []
        for i in range(5):
            question = pack[i][0]
            question_list.append(question)
            kg = str(pack[i][1]).replace('{"','{').replace('": "',': ').replace('", "',', ').replace('"}"','}').rsplit(',',1)[0]+"]"
            kg_list.append(kg)
            user_input = prefix + kg + ", " + question
            message = [{"role": "system", "content": "You answer the question based on your knowledge, with the given information for annotation, following the given format. Use [NA] for claims that need annotation but is unprovided."},
                        {"role": "user", "content": prompt_input},
                        {"role": "assistant", "content": eg_answer},
                        {"role": "user", "content": user_input}]
            message_list.append(message)
        output = ["","","","",""]
        for a in range(args.rerun): # re-run a few times if reach openai rate limit
            try:
                output = chatgpt.generate(message_list)
                break
            except:
                continue

        outputs = outputs + output

    chatgpt.save_output(question_list, kg_list, outputs)


if __name__ == "__main__":
    main(1)
    main(2)
    main(3)