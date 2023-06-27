from typing import Optional, List, Tuple

from torch.cuda import get_device_properties

from modules.device import torch_gc
from modules.options import cmd_opts

tokenizer = None
model = None


def prepare_model():
    global model
    if cmd_opts.cpu:
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

    model = model.eval()


def load_model():
    if cmd_opts.ui_dev:
        return

    from transformers import AutoModel, AutoTokenizer

    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(cmd_opts.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(cmd_opts.model_path, trust_remote_code=True)
    prepare_model()


def infer(query,
          history: Optional[List[Tuple]],
          max_length, top_p, temperature, use_stream_chat: bool):
    if cmd_opts.ui_dev:
        import time
        while True:
          yield query, "hello, dev mode %s" % time.ctime()
          time.sleep(1)

    if not model:
        raise "模型未加载"

    if history is None:
        history = []

    output_pos = 0
    try:
        print('-' * 50)
        print(str(query))
        print('=' * 50)
        if use_stream_chat:
            for output, history in model.stream_chat(
                    tokenizer, query=query, history=history,
                    max_length=max_length,
                    top_p=top_p,
                    temperature=temperature
            ):
                print(output[output_pos:], end='', flush=True)
                output_pos = len(output)
                yield query, output

        else:
            output, history = model.chat(
                tokenizer, query=query, history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature
            )

            print(output, end='')
            yield query, output

    except Exception as e:
        print("")
        print('*' * 50)
        print(f"生成失败: {repr(e)}", end='')

    print()
    torch_gc()
