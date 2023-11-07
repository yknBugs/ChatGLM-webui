# ChatGLM-webui

A webui for ChatGLM made by THUDM. [chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)

![image](https://user-images.githubusercontent.com/36563862/226985330-48e3b7f8-8c03-4778-af39-fd9b3a993d19.png)

## Features

- Original Chat like [chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)'s demo, but use Gradio Chatbox for better user experience.
- One click install script (but you still must install python)
- More parameters that can be freely adjusted
- Convenient save/load dialog history, presets
- Custom maximum context length
- Save to Markdown
- Use program arguments to specify model and caculation accuracy

## Install

### requirements

python3.10

```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade -r requirements.txt
```

or

```shell
bash install.sh
```

## Run

```shell
python webui.py
```

### Arguments

`--model-path`: specify model path. If this parameter is not specified manually, the default value is `THUDM/chatglm-6b`. Transformers will automatically download model from huggingface.

`--model`: specify model version. chatglm3(default), chatglm2, chatglm

`--listen`: launch gradio with 0.0.0.0 as server name, allowing to respond to network requests

`--autolaunch`: open url in web browser after launching webui

`--port`: webui port

`--share`: use gradio to share

`--device-id`: select the default CUDA device to use

`--precision`: fp32(CPU only), fp16, int4(CUDA GPU only), int8(CUDA GPU only)

`--cpu`: use cpu

`--path-prefix`: url root path. If this parameter is not specified manually, the default value is `/`. Using a path prefix of `/foo/bar` enables ChatGLM-webui to serve from `http://$ip:$port/foo/bar/` rather than `http://$ip:$port/`.
