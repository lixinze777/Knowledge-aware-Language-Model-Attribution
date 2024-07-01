import logging
logger = logging.getLogger(__name__)
import argparse
import openai
from tqdm import tqdm
import json
import asyncio
import csv
from typing import Any

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type = str, default = "../prompt/answer_question_simple_prompt.json", help = "Path to the prompt file")
    parser.add_argument("--input_path", type = str, default = "simple_question_graph.json", help = "Path to the input file")
    parser.add_argument("--output_path", type = str, default = "../outputs/gpt4_temp05_simple_1009.csv", help = "Path to the output file")
    parser.add_argument("--temperature", type=float, default=0.5, help= "Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help= "Nucleus sampling top-p")
    parser.add_argument("--model", type=str, default = "gpt-4-0613", help= "Model to use")
    parser.add_argument("--rerun", type=int, default = 3, help = "number of times to re-run the query when reach openai api rate limit")
    args = parser.parse_args()

    chatgpt = ChatGPT(args)
    questions = json.load(open(args.input_path))
    example = json.load(open(args.prompt_path))

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
            kg = str(pack[i][1]).replace("{'","{").replace("': '",": ").replace("', '",", ").replace("'}","}")
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
    main()
