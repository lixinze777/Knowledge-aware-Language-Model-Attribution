import logging
logger = logging.getLogger(__name__)
import argparse
import openai
from tqdm import tqdm
import json
import asyncio
import csv
from typing import Any
from statistics import mean, stdev
import re

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


    def save_output(self, question_list, kg_list, answer_list, outputs, ave):
        rows = zip(question_list, kg_list, answer_list, outputs)
        with open(self.args.output_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(["Question","Knowledge Graph","Answer", "Score"])
            for row in rows:
                writer.writerow(row)    


def main(seq, metric, file_name, folder):

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type = str, default = "../prompt/gpteval_prompt.json", help = "Path to the prompt file")
    parser.add_argument("--input_path", type = str, default = "../outputs/"+file_name+".csv", help = "Path to the input file")
    parser.add_argument("--output_path", type = str, default = "eval_results/"+folder+"/text_"+metric+".csv", help = "Path to the output file")
    parser.add_argument("--temperature", type=float, default=0, help= "Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help= "Nucleus sampling top-p")
    parser.add_argument("--model", type=str, default = "gpt-3.5-turbo-0613", help= "Model to use")
    parser.add_argument("--rerun", type=int, default = 3, help = "number of times to re-run the query when reach openai api rate limit")
    args = parser.parse_args()

    chatgpt = ChatGPT(args)
    inputs = []
    counter = 0
    with open(args.input_path) as f:
        for row in csv.reader(f):
            if row[1] != "Knowledge Graph" and row[1] != "[]" and counter < 100:
                inputs.append(row)
                counter += 1

    prompt = json.load(open(args.prompt_path))[seq]

    outputs = []
    question_list = []
    kg_list = []
    answer_list = []
    packs = [inputs[x:x+5] for x in range(0, len(inputs), 5)] # pack questions to packs of 5

    logger.info("----Initializing Generation----")
    for pack in tqdm(packs):
        message_list = []
        for i in range(5):
            question = pack[i][0]
            question_list.append(question)
            kg = str(pack[i][1]).replace("{'","{").replace("': '",": ").replace("', '",", ").replace("'}","}")
            kg_list.append(kg)
            answer = re.sub("[\[].*?[\]]", "", pack[i][2]).replace("  "," ").replace(" .", ".")
            answer_list.append(answer)
            message = [{"role": "system", "content": prompt + "Response is only one integer."},
                        {"role": "user", "content": "Question: %s/nKnowledge: %s/nAnswer: %s"%(question, kg, answer)},]
            message_list.append(message)
        output = ["","","","",""]
        for a in range(args.rerun): # re-run a few times if reach openai rate limit
            try:
                output = chatgpt.generate(message_list)
                break
            except:
                continue

        outputs = outputs + output
    _scores = []
    for output in outputs:
        if len(output) > 20:
            output = output[-10:]
        _score = 3
        try:
            _score = int(re.search('[0-9]+', output).group())
            assert _score <= 5
        except:
            print(output)
        _scores.append(_score)
    ave = mean(_scores)
    chatgpt.save_output(question_list, kg_list, answer_list, outputs, ave)
    return ave


def calculate_avg_std(num_list):
    return mean(num_list), stdev(num_list)

if __name__ == "__main__":
    metrics = ["coherence","consistency","fluency","relevance"]

    folder = "gpt4"

    file_names = [
        "gpt4_temp05_simple_1010"
    ]

    final_result = []
    for i in range(len(metrics)):
        scores = []
        for file_name in file_names:
            ave = main(i, metrics[i], file_name, folder)
            scores.append(ave)
            scores.append(ave)
            print(metrics[i]+": "+str(ave))
        _mean, _stdev = calculate_avg_std(scores)
        final_result.append([metrics[i], _mean, _stdev])

    with open('eval_results/'+folder+'/text_eval.txt', 'w') as fp:
        for item in final_result:
            fp.write("%s:  mean: %s   stdev: %s\n" % (item[0], str(item[1]), str(item[2])))
        