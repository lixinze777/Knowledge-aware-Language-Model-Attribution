import os
import time
import torch
import json
import argparse
import logging
from tqdm import tqdm
import torch
import numpy as np
from random import seed
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from easydict import EasyDict as edict
import csv

logger = logging.getLogger(__name__)


def generate_response(model, tokenizer, prompt, device, max_tokens, temperature=0.5):

    if max_tokens == 0:
        logger.warning("Prompt exceeds max length and return an empty string as answer.")
        return ""
    if max_tokens < 100:
        logger.warning("The model can at most generate < 100 tokens.")

    # `temperature` has to be a strictly positive float
    if temperature<=0.0:
        do_sample = False
        temperature += 0.1
    stop_word_id = 13 # '\n'

    batch_response = list()
    for inputs in prompt:
        inputs = tokenizer(inputs, return_tensors="pt")
        input_ids_length = inputs.input_ids.size(1)
        generate_ids = model.generate(
            inputs.input_ids.to(device), 
            do_sample=do_sample,
            temperature=temperature,
            eos_token_id=stop_word_id,
            max_new_tokens=max_tokens,
            num_return_sequences=1
        )
        response = tokenizer.batch_decode(generate_ids[:, input_ids_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        batch_response.append(
            edict(text=response.rstrip('\n'))
        )
    return batch_response


def save_output(question_list, kg_list, outputs, output_path):
    rows = zip(question_list, kg_list, outputs)
    with open(output_path+"/inference.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Question","Knowledge Graph","Answer"])
        for row in rows:
            writer.writerow(row)


def set_seed(args):
    if isinstance(args, int):
        seed(args)
        np.random.seed(args)
        torch.manual_seed(args)
    else:
        seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", default=0, type=int)
    parser.add_argument("--seed", default=43, type=int)
    parser.add_argument("--repeat_time", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--prompt_path", type = str, default = "../prompt/generate_prompt_sentlevel.json", help = "Path to the prompt file")
    parser.add_argument("--input_path", type = str, default = "../rerank/questions_selected_graphs.json", help = "Path to the input file")
    parser.add_argument("--output_path", type = str, default = "../outputs/alpaca_7b_0616", help = "Path to the output file")
    parser.add_argument("--model_name", default="../open_LLM/alpaca-7b", choices=["../open_LLM/alpaca-7b", "../open_LLM/vicuna-13b", "../open_LLM/LLaMA/llama-13b", "../open_LLM/LLaMA/llama-7b"], type=str)
    parser.add_argument("--batch_size_ICL", default=5, type=int)
    args = parser.parse_args()
    
    set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    logging.basicConfig(
        filename=os.path.join(args.output_path, "log_idx{}.txt".format(args.idx)), \
        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
    )
    logger.info(args)
    
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    questions = json.load(open(args.input_path))
    example = json.load(open(args.prompt_path))
    postfix = " Use [NA] for claims that need annotation but is unprovided."

    eg_question = example[0]
    eg_answer = example[1]
    eg_kg = str(example[2]).replace("{'","{").replace("': '",": ").replace("', '",", ").replace("'}","}")
    prompt1 = "Considering the information: "
    prompt_kg = eg_kg+", "
    prompt2 = eg_question + postfix + "\n"
    prompt3 = "Answer: "+ eg_answer+"\n"
    prompt = prompt1 + prompt_kg + prompt2 + prompt3

    outputs = []
    question_list = []
    kg_list = []
    assert(len(questions)==1085) # check data length
    batches = [questions[x:x+args.batch_size_ICL] for x in range(0, len(questions), args.batch_size_ICL)] # pack questions to batches length of 5
    
    time1 = time.time()
    for batch in tqdm(batches):
        batch_message_list = []
        for i in range(args.batch_size_ICL):
            question = batch[i][0]
            question_list.append(question)
            kg = str(batch[i][1]).replace("{'","{").replace("': '",": ").replace("', '",", ").replace("'}","}")
            kg_list.append(kg)
            prefix = "Considering the information: "
            prefix = prefix + kg + ", "
            message = prompt + prefix + question + postfix + "\n"
            batch_message_list.append(message)
        prompt_len = len(tokenizer.tokenize(max(batch_message_list, key=len)))
        print(args.max_length-prompt_len)
        output = generate_response(
            model, 
            tokenizer,
            batch_message_list,
            args.device,
            min(args.max_new_tokens, args.max_length-prompt_len),
            args.temperature, 
        )

        outputs = outputs + output
                
    time2 = time.time()
    logger.info("The time of executing evaluation: {}".format(time2-time1))

    save_output(question_list, kg_list, outputs, args.output_path)

