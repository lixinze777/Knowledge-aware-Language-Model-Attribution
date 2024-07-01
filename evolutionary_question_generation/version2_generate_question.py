import openai
import json
from tqdm import tqdm
import re
import asyncio
from typing import Any
import time

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


def generate_question(type, prompt, plain_text_list):
    prompt_paragraph = prompt[0]
    if type == 0: #simple question:
        prompt_question = prompt[2]
    else:
        prompt_question = prompt[1]
    outputs = []
    predictions = asyncio.run(
        dispatch_openai_requests(
	            messages_list=[
	            [{"role": "system", "content": "You are a question generator. Your objective is to generate one question such that the given paragraph is a good answer for it."},
	            {"role": "user", "content": "paragraph: "+prompt_paragraph},
	            {"role": "assistant", "content": "question: "+prompt_question},
	            {"role": "user", "content": "paragraph: "+plain_text_list[0]}],
	            [{"role": "system", "content": "You are a question generator. Your objective is to generate one question such that the given paragraph is a good answer for it."},
	            {"role": "user", "content": "paragraph: "+prompt_paragraph},
	            {"role": "assistant", "content": "question: "+prompt_question},
	            {"role": "user", "content": "paragraph: "+plain_text_list[1]}],
	            [{"role": "system", "content": "You are a question generator. Your objective is to generate one question such that the given paragraph is a good answer for it."},
	            {"role": "user", "content": "paragraph: "+prompt_paragraph},
	            {"role": "assistant", "content": "question: "+prompt_question},
	            {"role": "user", "content": "paragraph: "+plain_text_list[2]}],	[{"role": "system", "content": "You are a question generator. Your objective is to generate one question such that the given paragraph is a good answer for it."},
	            {"role": "user", "content": "paragraph: "+prompt_paragraph},
	            {"role": "assistant", "content": "question: "+prompt_question},
	            {"role": "user", "content": "paragraph: "+plain_text_list[3]}],
	            [{"role": "system", "content": "You are a question generator. Your objective is to generate one question such that the given paragraph is a good answer for it."},
	            {"role": "user", "content": "paragraph: "+prompt_paragraph},
	            {"role": "assistant", "content": "question: "+prompt_question},
	            {"role": "user", "content": "paragraph: "+plain_text_list[4]}],           
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


if __name__ == "__main__":

    data  = json.load(open("round5_extended_paragraph_201_1085.json"))
    prompt = json.load(open("prompt/generated_question_prompt.json"))
    batchs = [data[x:x+5] for x in range(0, len(data), 5)] # pack data to batchs of 5
    simple_outputs = []
    complex_outputs = []

    for batch in tqdm(batchs):
        labeled_text_list = []
        plain_text_list = []
        labeled_text_list = []
        for i in range(5):
            labeled_text = batch[i][0]
            plain_text = re.sub("[\[].*?[\]]", "", labeled_text)

            labeled_text_list.append(labeled_text)
            plain_text_list.append(plain_text)

        simple_question_list = generate_question(0, prompt, plain_text_list) # 0 is simple
        time.sleep(2)
        complex_question_list = generate_question(1, prompt, plain_text_list) # 1 is complex
        time.sleep(2)

        for zk in range(3): # re-run a few times if reach openai rate limit
            try:
                for i in range(5):
                    simple_outputs.append([labeled_text_list[i], simple_question_list[i]])
                    complex_outputs.append([labeled_text_list[i], complex_question_list[i]])
                break
            except:
                print("error occured")

    with open('generated_questions_simple_201_1085.json', 'w') as f:
        json.dump(simple_outputs, f)

    with open('generated_questions_complex_201_1085.json', 'w') as f:
        json.dump(complex_outputs, f)




