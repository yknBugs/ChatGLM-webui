import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--port", type=int, default="17860")
parser.add_argument("--model-path", type=str, default="THUDM/chatglm-6b")
parser.add_argument("--pre_seq_len", type=int, default=128)
parser.add_argument("--prefix_projection", type=bool, default=False)
parser.add_argument("--ptuning_checkpoint", type=str, default=None)
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["fp32", "fp16", "int4", "int8", "mps"])
parser.add_argument("--model", type=str, help="select chatglm model version", choices=["chatglm", "chatglm2", "chatglm3"])
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--autolaunch", action='store_true', help="automatically open url in web browser after launching")
parser.add_argument("--cpu", action='store_true', help="use cpu")
parser.add_argument("--share", action='store_true', help="use gradio share")
parser.add_argument("--device-id", type=str, help="select the default CUDA device to use", default=None)
parser.add_argument("--ui-dev", action='store_true', help="ui develop mode", default=None)
parser.add_argument("--path-prefix", type=str, help="url root path, e.g. /app", default="")

cmd_opts = parser.parse_args()
need_restart = False
