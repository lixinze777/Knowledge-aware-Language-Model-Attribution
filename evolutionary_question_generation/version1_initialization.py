import openai
import json
from tqdm import tqdm
import random
import asyncio
import csv
from typing import Any

openai.api_key = "Your Key"

async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)



def init_question(prompt, text_list, kg_list):
    prompt_question = prompt[0]
    prompt_answer = prompt[1]
    prompt_anno = prompt[2]
    prompt_kg = prompt[3]

    predictions = asyncio.run(
        dispatch_openai_requests(
            messages_list=[
            [{"role": "system", "content": "I want to you act as a Question Generator. Your objective is to select relevant knowledge to label the sentence and generate a question"},
            {"role": "user", "content": "sentence: "+prompt_answer+" knowledge: " + str(prompt_kg)},
            {"role": "assistant", "content": "sentence: "+prompt_anno+" question: "+prompt_question},
            {"role": "user", "content": "sentence: "+text_list[0]+" knowledge: " + str(kg_list[0])}],
            [{"role": "system", "content": "I want to you act as a Question Generator. Your objective is to select relevant knowledge to label the sentence and generate a question"},
            {"role": "user", "content": "sentence: "+prompt_answer+" knowledge: " + str(prompt_kg)},
            {"role": "assistant", "content": "sentence: "+prompt_anno+" question: "+prompt_question},
            {"role": "user", "content": "sentence: "+text_list[1]+" knowledge: " + str(kg_list[1])}],
            [{"role": "system", "content": "I want to you act as a Question Generator. Your objective is to select relevant knowledge to label the sentence and generate a question"},
            {"role": "user", "content": "sentence: "+prompt_answer+" knowledge: " + str(prompt_kg)},
            {"role": "assistant", "content": "sentence: "+prompt_anno+" question: "+prompt_question},
            {"role": "user", "content": "sentence: "+text_list[2]+" knowledge: " + str(kg_list[2])}],
            [{"role": "system", "content": "I want to you act as a Question Generator. Your objective is to select relevant knowledge to label the sentence and generate a question"},
            {"role": "user", "content": "sentence: "+prompt_answer+" knowledge: " + str(prompt_kg)},
            {"role": "assistant", "content": "sentence: "+prompt_anno+" question: "+prompt_question},
            {"role": "user", "content": "sentence: "+text_list[3]+" knowledge: " + str(kg_list[3])}],
            [{"role": "system", "content": "I want to you act as a Question Generator. Your objective is to select relevant knowledge to label the sentence and generate a question"},
            {"role": "user", "content": "sentence: "+prompt_answer+" knowledge: " + str(prompt_kg)},
            {"role": "assistant", "content": "sentence: "+prompt_anno+" question: "+prompt_question},
            {"role": "user", "content": "sentence: "+text_list[4]+" knowledge: " + str(kg_list[4])}],
            ],
            model="gpt-3.5-turbo",
            temperature=0,
            top_p=1.0,
        )
    )

    outputs = []
    for i, x in enumerate(predictions):
        output = x['choices'][0]['message']['content']
        outputs.append(output)

    return outputs


def process_output(text):
    text = text.replace("Question:","question:").replace("Sentence:","sentence:")
    if "question:" in text:
        answer = text.split("question:")[0][10:]
        question = text.split("question:")[1][1:]
        error = ""
    else:
        answer = ""
        question = ""
        error = text
    return [answer, question, error]


if __name__ == "__main__":
    prompt = json.load(open("prompt/turbo_init_prompt.json"))
    keys_to_pop = ['instance of', 'mother', 'father', 'genre', 'sex or gender', 'surname', 'child', 'given name', 'country of citizenship', 'country','described by source']
    data = []
    outputs = []
    with open('basic_info.csv') as f:
        for row in csv.reader(f):
            data.append(row)
    data = data[1:] # remove header
    assert(len(data)==1085) # check data length
    packs = [data[x:x+5] for x in range(0, len(data), 5)] # pack data to packs of 5

    for pack in tqdm(packs[:2]):
        text_list = []
        kg_list = []
        for i in range(5):
            text_list.append(pack[i][3])
            kg_list.append(json.loads(pack[i][2]))
        for a in range(0,3): # re-run 3 times if reach openai rate limit
            try:
                output = init_question(prompt, text_list, kg_list)
                break
            except:
                continue
        outputs = outputs + output

    results = []
    for i in range(len(outputs)):
        results.append(data[i]+process_output(outputs[i]))

    with open("outputs/round1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["name1","name2","knowledge_graph","example","answer","question","error"])
        for row in results:
            writer.writerow(row)
