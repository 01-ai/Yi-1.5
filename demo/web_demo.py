import gradio as gr
from argparse import ArgumentParser
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(next(model.parameters()).device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_tokens,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": 1.2,
        "eos_token_id": [tokenizer.eos_token_id],
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    response = ""
    for new_token in streamer:
        if new_token != "":
            response += new_token
            yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default="01-ai/Yi-1.5-34B-Chat",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=True,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, device_map="auto", torch_dtype="auto"
    ).eval()

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=args.inbrowser,
        share=args.share,
    )
