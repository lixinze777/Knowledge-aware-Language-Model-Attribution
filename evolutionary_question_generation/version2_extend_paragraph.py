import openai
import json
from tqdm import tqdm
import re
import asyncio
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

'''
def calculate_perplexity(sentence):
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()
'''

def extend_text(prompt, labeled_text_list, knowledge_list):
    original_paragraph = prompt[0]
    extend_answer = prompt[1]
    prompt_knowledge = prompt[2]
    outputs = []

    predictions = asyncio.run(
        dispatch_openai_requests(
            messages_list=[
                [{"role": "system", "content": "Your objective is to extend the original paragraph by adding one sentence that includes the given knowledge."},
                {"role": "user", "content": "paragraph: "+original_paragraph+" knowledge: "+prompt_knowledge},
                {"role": "assistant", "content": "answer: "+extend_answer},
                {"role": "user", "content": "paragraph: "+labeled_text_list[0]+" knowledge: "+knowledge_list[0]}],
                [{"role": "system", "content": "Your objective is to extend the original paragraph by adding one sentence that includes the given knowledge."},
                {"role": "user", "content": "paragraph: "+original_paragraph+" knowledge: "+prompt_knowledge},
                {"role": "assistant", "content": "answer: "+extend_answer},
                {"role": "user", "content": "paragraph: "+labeled_text_list[1]+" knowledge: "+knowledge_list[1]}],
                [{"role": "system", "content": "Your objective is to extend the original paragraph by adding one sentence that includes the given knowledge."},
                {"role": "user", "content": "paragraph: "+original_paragraph+" knowledge: "+prompt_knowledge},
                {"role": "assistant", "content": "answer: "+extend_answer},
                {"role": "user", "content": "paragraph: "+labeled_text_list[2]+" knowledge: "+knowledge_list[2]}],
                [{"role": "system", "content": "Your objective is to extend the original paragraph by adding one sentence that includes the given knowledge."},
                {"role": "user", "content": "paragraph: "+original_paragraph+" knowledge: "+prompt_knowledge},
                {"role": "assistant", "content": "answer: "+extend_answer},
                {"role": "user", "content": "paragraph: "+labeled_text_list[3]+" knowledge: "+knowledge_list[3]}],
                [{"role": "system", "content": "Your objective is to extend the original paragraph by adding one sentence that includes the given knowledge."},
                {"role": "user", "content": "paragraph: "+original_paragraph+" knowledge: "+prompt_knowledge},
                {"role": "assistant", "content": "answer: "+extend_answer},
                {"role": "user", "content": "paragraph: "+labeled_text_list[4]+" knowledge: "+knowledge_list[4]}],
            ],
            model="gpt-4-0613",
            temperature=0.1,
            top_p=1.0,
        )
    )

    for i, x in enumerate(predictions):
        output = x['choices'][0]['message']['content']
        outputs.append(output)

    return outputs



def score_function(scores, property, name_key_value, plain_text, perp_template):
    importance_score = 0
    try:
        importance_score = scores[property]
    except:
        pass
    coherence_score = 0
    '''
    name, key, value = name_key_value
    if key in perp_template[0]:
        sent = name +" is "+key+" "+value+"."
    elif key in perp_template[1]:
        sent = name+" "+key+" "+value+"."
    else:
        sent = name+"'s "+key+" is "+value+"."
    coherence_score = 1/calculate_perplexity(plain_text+sent)
    '''
    overall_score = importance_score + coherence_score
    return overall_score


if __name__ == "__main__":

    keys_to_pop = ['instance of', 'genre', 'sex or gender', 'surname', 'given name', 'country of citizenship', 'country','described by source']

    #data  = json.load(open("text_kg_list_20230918.json"))
    data  = json.load(open("round4_extended_paragraph_201_1085.json"))
    scores = json.load(open("property_scores.json"))
    extend_prompt = json.load(open("prompt/extend_prompt.json"))
    perp_template = json.load(open("perp_template.json"))
    outputs = []

    batchs = [data[x:x+5] for x in range(0, len(data), 5)] # pack data to batchs of 5

    for batch in tqdm(batchs):
        labeled_text_list = []
        knowledge_list = []
        person1_list = []
        person2_list = []
        for i in range(5):
            labeled_text = batch[i][0]
            plain_text = re.sub("[\[].*?[\]]", "", labeled_text)
            labels = re.findall(r'\[.*?\]', labeled_text)

            person1 = batch[i][1]
            person2 = batch[i][2]

            for item in person1:
                if ":" in item and item[:4] != "name": # not considering QID and name
                    for k in keys_to_pop:
                        if k in item:
                            person1.remove(item)
                            break
                    for l in labels:
                        if item in l and item in person1:
                            person1.remove(item)
                            continue

            for item in person2:
                if ":" in item and item[:4] != "name": # not considering QID and name
                    for k in keys_to_pop:
                        if k in item:
                            person2.remove(item)
                            break
                    for l in labels:
                        if item in l and item in person2:
                            person2.remove(item)
                            continue

            knowledge_to_select = []
            for item in person1:
                if item[:4] == "name":
                    name = item[6:]
                else:
                    if ":" not in item:
                        qid = item
                    else:
                        key = item.split(": ",1)[0]
                        value = item.split(": ",1)[1]
                        knowledge_to_select.append((key, [name,key,value], "["+qid+", "+name+", "+item+"]"))

            for item in person2:
                if item[:4] == "name":
                    name = item[6:]
                else:
                    if ":" not in item:
                        qid = item
                    else:
                        key = item.split(": ",1)[0]
                        value = item.split(": ",1)[1]
                        knowledge_to_select.append((key, [name,key,value], "["+qid+", "+name+", "+item+"]"))

            highest_score = 0
            selected_knowledge = ""
            for knowledge in knowledge_to_select:
                property = knowledge[0]
                name_key_value = knowledge[1]
                insert_knowledge = knowledge[2]
                score = score_function(scores, property, name_key_value, plain_text, perp_template)
                if score > highest_score:
                    highest_score = score
                    selected_knowledge = insert_knowledge

            labeled_text_list.append(labeled_text)
            knowledge_list.append(selected_knowledge)
            person1_list.append(person1)
            person2_list.append(person2)

        for zk in range(3): # re-run a few times if reach openai rate limit
            try:
                extended_text = extend_text(extend_prompt, labeled_text_list, knowledge_list)
                for i in range(5):
                    outputs.append([extended_text[i],person1_list[i], person2_list[i]])
                break
            except:
                print("error occured")

    with open('round5_extended_paragraph_201_1085.json', 'w') as f:
        json.dump(outputs, f)


