import openai
import json
from tqdm import tqdm
import random
import asyncio
from typing import Any
import csv

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


def extend_question(prompt, question_list, answer_list, knowledge_list):
    original_question = prompt[0]
    extend_question = prompt[1]
    original_answer = prompt[2]
    extend_answer = prompt[3]
    prompt_knowledge = prompt[4]
    outputs = []

    predictions = asyncio.run(
        dispatch_openai_requests(
            messages_list=[
                [{"role": "system", "content": "I want to you act as a Question Rewriter. Your objective is to extend the question and answer, so the they include the additional knowledge. There should be only one question, and use \"question:\" before it to indicate the question."},
                {"role": "user", "content": "answer: "+original_answer+" question: "+original_question+" knowledge: "+prompt_knowledge},
                {"role": "assistant", "content": "answer: "+extend_answer+" question: "+extend_question},
                {"role": "user", "content": "answer: "+answer_list[0]+" question: "+question_list[0]+" knowledge: "+knowledge_list[0]}],
                [{"role": "system", "content": "I want to you act as a Question Rewriter. Your objective is to extend the question and answer, so the they include the additional knowledge. There should be only one question, and use \"question:\" before it to indicate the question."},
                {"role": "user", "content": "answer: "+original_answer+" question: "+original_question+" knowledge: "+prompt_knowledge},
                {"role": "assistant", "content": "answer: "+extend_answer+" question: "+extend_question},
                {"role": "user", "content": "answer: "+answer_list[1]+" question: "+question_list[1]+" knowledge: "+knowledge_list[1]}],
                [{"role": "system", "content": "I want to you act as a Question Rewriter. Your objective is to extend the question and answer, so the they include the additional knowledge. There should be only one question, and use \"question:\" before it to indicate the question."},
                {"role": "user", "content": "answer: "+original_answer+" question: "+original_question+" knowledge: "+prompt_knowledge},
                {"role": "assistant", "content": "answer: "+extend_answer+" question: "+extend_question},
                {"role": "user", "content": "answer: "+answer_list[2]+" question: "+question_list[2]+" knowledge: "+knowledge_list[2]}],
                [{"role": "system", "content": "I want to you act as a Question Rewriter. Your objective is to extend the question and answer, so the they include the additional knowledge. There should be only one question, and use \"question:\" before it to indicate the question."},
                {"role": "user", "content": "answer: "+original_answer+" question: "+original_question+" knowledge: "+prompt_knowledge},
                {"role": "assistant", "content": "answer: "+extend_answer+" question: "+extend_question},
                {"role": "user", "content": "answer: "+answer_list[3]+" question: "+question_list[3]+" knowledge: "+knowledge_list[3]}],
                [{"role": "system", "content": "I want to you act as a Question Rewriter. Your objective is to extend the question and answer, so the they include the additional knowledge. There should be only one question, and use \"question:\" before it to indicate the question."},
                {"role": "user", "content": "answer: "+original_answer+" question: "+original_question+" knowledge: "+prompt_knowledge},
                {"role": "assistant", "content": "answer: "+extend_answer+" question: "+extend_question},
                {"role": "user", "content": "answer: "+answer_list[4]+" question: "+question_list[4]+" knowledge: "+knowledge_list[4]}],
            ],
            model="gpt-3.5-turbo",
            temperature=0.1,
            top_p=1.0,
        )
    )

    for i, x in enumerate(predictions):
        output = x['choices'][0]['message']['content']
        outputs.append(output)

    return outputs


def process_output(text):
    text = text.replace("Question:","question:").replace("Sentence:","sentence:").replace("Knowledge:","knowledge")
    if "question:" in text:
        answer = text.split("question:")[0][8:]
        answer = answer.split("knowledge:")[0]
        question = text.split("question:")[1][1:]
        question = question.split("knowledge:")[0]
        error = ""
    elif text[-1:] == "?":
        answer = text.rsplit(".",1)[0][8:]
        answer = answer.split("knowledge:")[0]
        question = text.rsplit(".",1)[1][1:]
        question = question.split("knowledge:")[0]
        error = ""
    else:
        answer = ""
        question = ""
        error = text
    return [question, answer, error]


if __name__ == "__main__":
    prompt = json.load(open("prompt/turbo_extend_prompt.json"))
    keys_to_pop = ['instance of', 'mother', 'father', 'genre', 'sex or gender', 'surname', 'child', 'given name', 'country of citizenship', 'country','described by source']
    data = []
    outputs = []
    with open('outputs/round1.csv') as f:
        for row in csv.reader(f):
            data.append(row)
    data = data[1:] # remove header
    assert(len(data)==1085) # check data length
    packs = [data[x:x+5] for x in range(0, len(data), 5)] # pack data to packs of 5
    overall_knowledge_list = []
    overall_left_kg_list = []
    for pack in tqdm(packs):
        question_list = []
        answer_list = []
        knowledge_list = []
        kg_list = []

        for i in range(5):
            question = pack[i][4]
            question_list.append(question)
            answer_list.append(pack[i][5])
            kg = json.loads(pack[i][2])

            graph1 = kg[0]
            graph2 = kg[1]
            relation = kg[2]
            name1 = graph1["name"]
            name2 = graph2["name"]

            for k in keys_to_pop:
                if k in graph1:
                    graph1.pop(k)
                if k in graph2:
                    graph2.pop(k)

            if name1 in question:
                graph_temp = graph1.copy()
                graph_temp.pop("qid")
                graph_temp.pop("name")
                property, value = random.choice(list(graph_temp.items()))
                qid = graph1["qid"]
                knowledge = "[qid: "+str(qid)+", name: "+name1+", "+property+": "+value+"]"
                graph1.pop(property) # what's left
            else:
                graph_temp = graph2.copy()
                graph_temp.pop("qid")
                graph_temp.pop("name")
                property, value = random.choice(list(graph_temp.items()))
                qid = graph2["qid"]
                knowledge = "[qid: "+str(qid)+", name: "+name2+", "+property+": "+value+"]"
                graph2.pop(property) # what's left
            knowledge_list.append(json.dumps(knowledge))
            kg_list.append(json.dumps([graph1,graph2,relation]))
        o = []
        for a in range(0,3): # re-run 3 times if reach openai rate limit
            try:
                o = extend_question(prompt, question_list, answer_list, knowledge_list)
                break
            except:
                continue
        outputs = outputs + o       
        overall_knowledge_list = overall_knowledge_list + knowledge_list
        overall_left_kg_list = overall_left_kg_list + kg_list

    results = []
    for i in range(len(outputs)):
        results.append(data[i][:4]+[overall_knowledge_list[i]]+[overall_left_kg_list[i]]+process_output(outputs[i]))

    with open("outputs/round2.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["name1","name2","knowledge_graph","example","knowledge","left_kg","question","answer","error"])
        for row in results:
            writer.writerow(row)

