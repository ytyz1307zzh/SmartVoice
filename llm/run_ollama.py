import argparse
import requests
import ollama
import time
import json
from typing import Dict
from transformers import AutoTokenizer


def call_ollama_preload(generate_args: Dict, model: str, tokenizer: AutoTokenizer, prompt: str):
    '''
    Call an ollama model that has already been loaded to memory via `ollama run model_name` in a separate process
    Tokenizer has to be the corresponding huggingface tokenizer
    '''
    messages = [{
        "role": "user",
        "content": prompt
    }]

    messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": messages,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": generate_args["temperature"],
            "num_predict": generate_args["max_new_tokens"],
            "top_k": generate_args["top_k"],
            "top_p": generate_args["top_p"],
        },
    }

    headers = {
        'Content-Type': 'application/json'
    }

    start_time = time.time()
    response_body = requests.post(url, data=json.dumps(payload), headers=headers)
    response = response_body.json()['response'].strip()
    
    return response


def call_ollama_no_preload(generate_args: Dict, model: str, prompt: str):
    '''
    Call an ollama model without pre-loading it.
    Actually, I found that if this function was run once, then the created ollama process will remain running in the background,
    which works like a 'preload'.
    '''
    messages = [{
        "role": "user",
        "content": prompt
    }]

    start_time = time.time()
    response_body = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": generate_args["temperature"],
            "num_predict": generate_args["max_new_tokens"],
            "top_k": generate_args["top_k"],
            "top_p": generate_args["top_p"],
        },
    )

    response = response_body["message"]["content"].strip()

    return response
    

    
# The following code is for testing the above functions
'''
if __name__ == "__main__":
    generate_args = {
        "temperature": 0.0,
        "max_new_tokens": 300,
        "top_k": 1,
        "top_p": 0.0
    }
    model = "qwen2.5:0.5b-instruct-q2_K"
    prompt = "Repeat the following paragraph word by word: Qwen2.5 is the new series of Qwen large language models. A number of base language models and instruction-tuned models are released."
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    start_time = time.time()
    # response = call_ollama_preload(generate_args, model, tokenizer, prompt)
    response = call_ollama_no_preload(generate_args, model, prompt)
    print('THe whole function finished. Call time: ', round(time.time() - start_time, 2))
    print(response)
'''
    
