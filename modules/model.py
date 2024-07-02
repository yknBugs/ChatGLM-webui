import traceback

from threading import Thread
from torch.cuda import get_device_properties

from modules.device import torch_gc
from modules.options import cmd_opts

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import os
import torch

tokenizer = None
model = None


def get_device():
    if cmd_opts.cpu:
        return 'cpu'
    elif cmd_opts.precision == 'mps':
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def prepare_model():
    global model
    if get_device() == 'cpu':
        if cmd_opts.precision == "fp32":
            model = model.float()
        elif cmd_opts.precision == "bf16":
            model = model.bfloat16()
        else:
            model = model.float()
    else:
        if cmd_opts.precision is None:
            total_vram_in_gb = get_device_properties(0).total_memory / 1e9
            print(f'显存: {total_vram_in_gb:.2f} GB')

            if total_vram_in_gb > 30:
                cmd_opts.precision = 'fp32'
            elif total_vram_in_gb > 13:
                cmd_opts.precision = 'fp16'
            elif total_vram_in_gb > 10:
                cmd_opts.precision = 'int8'
            else:
                cmd_opts.precision = 'int4'

            print(f'根据你的显存容量，自动选择了精度 {cmd_opts.precision}'
                  f' 如果你需要自己选择精度，'
                  f' 请在启动时传入参数 --precision 来选择精度')

        if cmd_opts.precision == "fp16":
            model = model.half().cuda()
        elif cmd_opts.precision == "int4":
            model = model.half().quantize(4).cuda()
        elif cmd_opts.precision == "int8":
            model = model.half().quantize(8).cuda()
        elif cmd_opts.precision == "fp32":
            model = model.float()
        elif cmd_opts.precision == "mps":
            model = model.to('mps')

    model = model.eval()


def load_glm4_model():
    global model

    if get_device() == 'cpu':
        if cmd_opts.precision == "fp32":
            model = AutoModelForCausalLM.from_pretrained(cmd_opts.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).float()
            model = model.to('cpu')
        else:
            model = AutoModelForCausalLM.from_pretrained(cmd_opts.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
            model = model.to('cpu')
    else:
        if cmd_opts.precision is None:
            total_vram_in_gb = get_device_properties(0).total_memory / 1e9
            print(f'显存: {total_vram_in_gb:.2f} GB')

            if total_vram_in_gb > 16:
                cmd_opts.precision = 'bf16'
            elif total_vram_in_gb > 12:
                cmd_opts.precision = 'int8'
            else:
                cmd_opts.precision = 'int4'

            print(f'根据你的显存容量，自动选择了精度 {cmd_opts.precision}'
                  f' 如果你需要自己选择精度，'
                  f' 请在启动时传入参数 --precision 来选择精度')

        if cmd_opts.precision == "bf16":
            model = AutoModelForCausalLM.from_pretrained(cmd_opts.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
            model = model.to('cuda')
        elif cmd_opts.precision == "fp16":
            model = AutoModelForCausalLM.from_pretrained(cmd_opts.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).half()
            model = model.to('cuda')
        elif cmd_opts.precision == "int4":
            model = AutoModelForCausalLM.from_pretrained(cmd_opts.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, load_in_4bit=True ,device_map='cuda')
        elif cmd_opts.precision == "int8":
            model = AutoModelForCausalLM.from_pretrained(cmd_opts.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, load_in_8bit=True ,device_map='cuda')
        elif cmd_opts.precision == "fp32":
            model = AutoModelForCausalLM.from_pretrained(cmd_opts.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).float()
            model = model.to('cuda')
        elif cmd_opts.precision == "mps":
            model = AutoModelForCausalLM.from_pretrained(cmd_opts.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
            model = model.to('mps')


def load_model():
    if cmd_opts.ui_dev:
        return

    global tokenizer, model

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(cmd_opts.model_path, trust_remote_code=True)
    config.pre_seq_len = cmd_opts.pre_seq_len
    config.prefix_projection = cmd_opts.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(cmd_opts.model_path, trust_remote_code=True)
    if cmd_opts.model is None or cmd_opts.model == "chatglm4":
        load_glm4_model()
    else:
        model = AutoModel.from_pretrained(cmd_opts.model_path, config=config, trust_remote_code=True)

    if cmd_opts.ptuning_checkpoint is not None:
        # Load ptuning weights
        prefix_state_dict = torch.load(os.path.join(cmd_opts.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v

        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    if cmd_opts.model is None or cmd_opts.model == "chatglm4":
        model = model.eval()
    else:
        prepare_model()


def infer(query, history,
          max_length, top_p, temperature, use_stream_chat: bool):
    if cmd_opts.ui_dev:
        import time
        if cmd_opts.model is None or cmd_opts.model == "chatglm3" or cmd_opts.model == "chatglm4":
            history.append({'role': 'user', 'content': query})
            history.append({'role': 'assistant', 'metadata': '', 'content': "hello, dev mode %s" % time.ctime()})
        else:
            history.append((query, "hello, dev mode %s" % time.ctime()))
        while True:
            yield query, "hello, dev mode %s" % time.ctime(), history
            time.sleep(1)

    if not model:
        raise "模型未加载"

    if history is None:
        history = []

    #####
    #  ChatGLM2 以下的 History 返回格式
    #  [('你好', '你好！有什么我可以帮助你的吗'), ('你是谁', '我是一个大型语言模型')]
    #
    #  ChatGLM3 History 返回格式
    #  [{'role': 'user', 'content': '你好'}, 
    #   {'role': 'assistant', 'metadata': '', 'content': '你好！有什么我可以帮助你的吗？'}, 
    #   {'role': 'user', 'content': '你是谁'}, 
    #   {'role': 'assistant', 'metadata': '', 'content': '我是一个大型语言模型。'}]
    #####

    output_pos = 0
    try:
        with torch.no_grad():
            print('-' * 50)
            print(str(query))
            print('=' * 50)
            if use_stream_chat:
                if cmd_opts.model is None or cmd_opts.model == "chatglm4":
                    history.append({'role': 'user', 'content': query})
                    model_inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(get_device())
                    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
                    gen_kwargs = {
                        "input_ids": model_inputs, 
                        "streamer": streamer, 
                        "max_new_tokens": max_length, 
                        "do_sample": True, 
                        "top_p": top_p, 
                        "temperature": temperature, 
                        "repetition_penalty": 1.2
                    }
                    history.append({'role': 'assistant', 'content': ''})
                    t = Thread(target=model.generate, kwargs=gen_kwargs)
                    t.start()
                    output = ""
                    for new_token in streamer:
                        if new_token:
                            print(new_token, end='', flush=True)
                            output += new_token
                            history[-1]['content'] += new_token
                            yield query, output, history

                else:
                    for output, history in model.stream_chat(
                            tokenizer, query=query, history=history,
                            max_length=max_length,
                            top_p=top_p,
                            temperature=temperature
                    ):
                        print(output[output_pos:], end='', flush=True)
                        output_pos = len(output)
                        yield query, output, history
            else:
                if cmd_opts.model is None or cmd_opts.model == "chatglm4":
                    history.append({'role': 'user', 'content': query})
                    model_inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
                    model_inputs = model_inputs.to(get_device())
                    gen_kwargs = {
                        "max_length": max_length, 
                        "do_sample": True, 
                        "top_p": top_p, 
                        "temperature": temperature, 
                        "repetition_penalty": 1.2
                    }
                    output = model.generate(**model_inputs, **gen_kwargs)
                    output = output[:, model_inputs['input_ids'].shape[1]:]
                    output = tokenizer.decode(output[0], skip_special_tokens=True)
                    history.append({'role': 'assistant', 'content': output})
                else:
                    output, history = model.chat(
                        tokenizer, query=query, history=history,
                        max_length=max_length,
                        top_p=top_p,
                        temperature=temperature
                    )

                if cmd_opts.model is None or cmd_opts.model == "chatglm3" or cmd_opts.model == "chatglm4":
                    if history[-2]['role'] == 'assistant':
                        print(history[-2]['content'])
                        print('=' * 50)
                    print(history[-1]['content'], end='')
                else:
                    print(output, end='')
                yield query, output, history
    except Exception as e:
        print("")
        print('*' * 50)
        print(f"生成失败: {repr(e)}", end='')
        traceback.print_exc()

    print()
    torch_gc()
