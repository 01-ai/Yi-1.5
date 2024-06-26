<div align="center">

<picture> 
  <img src="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg" width="150px">
</picture>

</div>

<br/>

<p align="center">
  <a href="https://huggingface.co/01-ai">🤗 HuggingFace</a> •
  <a href="https://www.modelscope.cn/organization/01ai/">🤖 ModelScope</a> •
  <a href="https://wisemodel.cn/organization/01.AI">🟣 wisemodel</a> 
  <br/>
  <a href="https://discord.gg/hYUwWddeAu">👾 Discord</a> •
  <a href="https://twitter.com/01ai_yi">🐤 Twitter</a> •
  <a href="https://github.com/01-ai/Yi-1.5/issues/2">💬 WeChat</a> 
  <br/>
  <a href="https://arxiv.org/abs/2403.04652">📝 Paper</a> •
  <a href="https://01-ai.github.io/">💪 Tech Blog</a> •
  <a href="https://github.com/01-ai/Yi/tree/main?tab=readme-ov-file#faq">🙌 FAQ</a> •
  <a href="https://github.com/01-ai/Yi/tree/main?tab=readme-ov-file#learning-hub">📗 Learning Hub</a>
</p>

---

- [Intro](#intro)
- [News](#news)
- [Quick Start](#quick-start)
- [Web Demo](#web-demo)
- [Deployment](#deployment)
- [Fine-tuning](#fine-tuning)
- [API](#api)
- [License](#license)

## Intro

Yi-1.5 is an upgraded version of Yi. It is continuously pre-trained on Yi with a high-quality corpus of 500B tokens and fine-tuned on 3M diverse fine-tuning samples. 

Compared with Yi, Yi-1.5 delivers stronger performance in coding, math, reasoning, and instruction-following capability, while still maintaining excellent capabilities in language understanding, commonsense reasoning, and reading comprehension. 

Yi-1.5 comes in 3 model sizes: 34B, 9B, and 6B. For model details and benchmarks, see [Model Card](https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8).

## News

- 2024-05-13: The Yi-1.5 series models are open-sourced, further improving coding, math, reasoning, and instruction-following abilities. 

## Requirements

- Make sure Python 3.10 or a later version is installed.

- Set up the environment and install the required packages.

  ```bash
  pip install -r requirements.txt
  ```
  
- Download the Yi-1.5 model from [Hugging Face](https://huggingface.co/01-ai), [ModelScope](https://www.modelscope.cn/organization/01ai/), or [WiseModel](https://wisemodel.cn/organization/01.AI).

## Quick Start

This tutorial runs Yi-1.5-34B-Chat locally on an A800 (80G).

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
  
  input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
  output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id)
  response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
  
  # Model response: "Hello! How can I assist you today?"
  print(response)
  ```

### Ollama 




You can run Yi-1.5 models on Ollama locally.

1. After [installing Ollama](https://github.com/ollama/ollama/tree/main/docs), you can start the Ollama service. Note that keep this service running while you use Ollama.
  
    ```python
    ollama serve
    ```

2. Run Yi-1.5 models. For more Yi models supported by Ollama, see [Yi tags](https://ollama.com/library/yi/tags).
   
    ```python
    ollama run yi:v1.5
    ```

3. Chat with Yi-1.5 via OpenAI-compatible API. For more details on how to use Yi-1.5 via OpenAI API and REST API on Ollama, see [Ollama docs](https://github.com/ollama/ollama/tree/main/docs).

    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',  # required but ignored
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': 'What is your name',
            }
        ],
        model='yi:1.5',
    )
    ```

## Deployment

Prerequisites: Before deploying Yi-1.5 models, make sure you meet the [software and hardware requirements](https://github.com/01-ai/Yi/tree/main?tab=readme-ov-file#software-requirements). 

### vLLM

Prerequisites: Download the latest version of [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html).

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

## Web Demo

You can activate Yi-1.5-34B-Chat through the [huggingface chat ui](https://huggingface.co/chat/settings/01-ai/Yi-1.5-34B-Chat/) then experience it.

Or you can build it locally by yourself, as follows:
```
python demo/web_demo.py -c <your-model-path>
```

## Fine-tuning

You can use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Swift](https://github.com/modelscope/swift), [XTuner](https://github.com/InternLM/xtuner), and [Firefly](https://github.com/yangjianxin1/Firefly) for fine-tuning. These frameworks all support fine-tuning the Yi series models.

## API

Yi APIs are OpenAI-compatible and provided at [Yi Platform](https://platform.lingyiwanwu.com/). Sign up to get free tokens, and you can also pay-as-you-go at a competitive price. Additionally, Yi APIs are also deployed on [Replicate](https://replicate.com/search?query=01+ai) and [OpenRouter](https://openrouter.ai/models?q=01%20ai). 

## License

The code and weights of the Yi-1.5 series models are distributed under the [Apache 2.0 license](https://github.com/01-ai/Yi/blob/main/LICENSE).

If you create derivative works based on this model, please include the following attribution in your derivative works:

    This work is a derivative of [The Yi-1.5 Series Model You Base On] by 01.AI, used under the Apache 2.0 License.

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>
