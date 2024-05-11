<div align="center">

<picture> 
  <img alt="specify theme context for images" src="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg" width="150px">
</picture>

</div>

<br/>
<br/>

<p align="center">
  <a href="https://github.com/01-ai">🤗 Hugging Face</a> •
  <a href="https://discord.gg/hYUwWddeAu">👾 Discord</a> •
  <a href="https://twitter.com/01ai_yi">🐤 Twitter</a> •
  <a href="https://github.com/01-ai/Yi/issues/43#issuecomment-1827285245">💬 WeChat</a> •
  <a href="https://arxiv.org/abs/2403.04652">📝 Paper</a>
</p>

# Quick start

This tutorial guides you through every step of running **Yi-1.5-34B-Chat locally on an A800 (80G)** and then performing inference.
 
Prerequisites: 

- Make sure Python 3.10 or a later version is installed.

- Set up the environment and install the required packages.

  ```bash
  git clone https://github.com/01-ai/Yi.git
  cd yi
  pip install -r requirements.txt
  ```

- Download the Yi-1.5 model from [Hugging Face](https://huggingface.co/01-ai), [ModelScope](https://www.modelscope.cn/organization/01ai/), or [WiseModel](https://wisemodel.cn/organization/01.AI).

## Chat models

Perform inference with Yi-1.5 chat models as below.

1. Create a file named  `quick_start.py` and copy the following content to it.

  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  
  model_path = '<your-model-path>'
  
  tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
  
  # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
  model = AutoModelForCausalLM.from_pretrained(
      model_path,
      device_map="auto",
      torch_dtype='auto'
  ).eval()
  
  # Prompt content: "hi"
  messages = [
      {"role": "user", "content": "hi"}
  ]
  
  input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
  output_ids = model.generate(input_ids.to('cuda'))
  response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
  
  # Model response: "Hello! How can I assist you today?"
  print(response)
  ```

 2. Run `quick_start.py`.

  ```bash
  python quick_start.py
  ```

  Then you can see an output similar to the one below. 🥳

  ```bash
  Hello! How can I assist you today?
  ```

## Base model

- Perform inference with Yi-1.5 base model

  The steps are similar to that of chat.

  You can use the existing file [`text_generation.py`](tbd).

  ```bash
  python demo/text_generation.py  --model <your-model-path>
  ```

  Then you can see an output similar to the one below. 🥳

  <details>

  <summary>Output. ⬇️ </summary>

  <br>

  **Prompt**: Let me tell you an interesting story about cat Tom and mouse Jerry,

  **Generation**: Let me tell you an interesting story about cat Tom and mouse Jerry, which happened in my childhood. My father had a big house with two cats living inside it to kill mice. One day when I was playing at home alone, I found one of the tomcats lying on his back near our kitchen door, looking very much like he wanted something from us but couldn’t get up because there were too many people around him! He kept trying for several minutes before finally giving up...

  </details>

- Yi-9B
  
  Input

  ```bash
  from transformers import AutoModelForCausalLM, AutoTokenizer

  MODEL_DIR = "01-ai/Yi-9B"
  model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

  input_text = "# write the quick sort algorithm"
  inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
  outputs = model.generate(**inputs, max_length=256)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  ```

  Output

  ```bash
  # write the quick sort algorithm
  def quick_sort(arr):
      if len(arr) <= 1:
          return arr
      pivot = arr[len(arr) // 2]
      left = [x for x in arr if x < pivot]
      middle = [x for x in arr if x == pivot]
      right = [x for x in arr if x > pivot]
      return quick_sort(left) + middle + quick_sort(right)

  # test the quick sort algorithm
  print(quick_sort([3, 6, 8, 10, 1, 2, 1]))
  ```


<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

ollama steps?

# Deployment

# Inference

## vLLM

Prerequisites: 

We advise you to use vLLM>=0.3.0 to build OpenAI-compatible API service. 

1. Start the server with a chat model.

```bash
python -m vllm.entrypoints.openai.api_server  --model 01-ai/Yi-1.5-9B-Chat  --served-model-name Yi-1.5-9B-Chat
```

2. Use the chat API.

  - HTTP

    ```bash
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Yi-1.5-9B-Chat",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"}
            ]
        }'
    ```

  - Python client

    ```python
    from openai import OpenAI
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    chat_response = client.chat.completions.create(
        model="Yi-1.5-9B-Chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    )
    print("Chat response:", chat_response)
    ```

## llama.cpp

# Web Demo

# FAQ

- Quantization: It is recommended to use AWQ, GPTQ, and GGUF to quantize Yi-1.5 models.
  
- Fine-tuning: It is recommended to use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune Yi-1.5 models with SFT, DPO, PPO, etc.

We advise you to use training frameworks, including Axolotl, Llama-Factory, Swift, etc., to finetune your models with SFT, DPO, PPO, etc.










