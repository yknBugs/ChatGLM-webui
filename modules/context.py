from typing import List, Tuple
import json
import os
import time
import traceback
from modules.options import cmd_opts

def parse_codeblock(text):
    if cmd_opts.model is None or cmd_opts.model == "chatglm3":
        lines = text.split("\n")
        lines = [line for line in lines if line != ""]
        count = 0
        for i, line in enumerate(lines):
            if "```" in line:
                count += 1
                items = line.split('`')
                if count % 2 == 1:
                    lines[i] = f'<pre><code class="language-{items[-1]}">'
                else:
                    lines[i] = f'<br></code></pre>'
            else:
                if i > 0:
                    if count % 2 == 1:
                        line = line.replace("`", "\`")
                        line = line.replace("<", "&lt;")
                        line = line.replace(">", "&gt;")
                        line = line.replace(" ", "&nbsp;")
                        line = line.replace("*", "&ast;")
                        line = line.replace("_", "&lowbar;")
                        line = line.replace("-", "&#45;")
                        line = line.replace(".", "&#46;")
                        line = line.replace("!", "&#33;")
                        line = line.replace("(", "&#40;")
                        line = line.replace(")", "&#41;")
                        line = line.replace("$", "&#36;")
                    lines[i] = "<br>"+line
        text = "".join(lines)
        return text
    else:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "```" in line:
                if line != "```":
                    lines[i] = f'<pre><code class="{lines[i][3:]}">'
                else:
                    lines[i] = '</code></pre>'
            else:
                if i > 0:
                    lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
        return "".join(lines)

STOPPED = 0
LOOP_FIRST = 1
LOOP = 2
INTERRUPTED = 3

class Context:
    def __init__(self, history = None):
        if history:
            self.history = history
        else:
            self.history = []
        self.rh = []
        self.state = STOPPED
        self.max_rounds = 25
        self.max_words = 8192

    def inferBegin(self):
        self.limit_round()
        self.limit_word()
        self.state = LOOP_FIRST

    def interrupt(self):
        if self.state == LOOP_FIRST or self.state == LOOP:
            self.state = INTERRUPTED

    def inferLoop(self, query, output, history) -> bool:
        # c: List[Tuple[str, str]]
        if self.state == INTERRUPTED:
            return True
        elif self.state == LOOP_FIRST:
            self.history = history
            self.rh.append((query, parse_codeblock(output)))
            self.state = LOOP
        else:
            self.history = history
            self.rh[-1] = (query, output)
        return False

    def inferEnd(self) -> None:
        if self.rh:
            query, output = self.rh[-1]
            self.rh[-1] = (query, parse_codeblock(output))
        self.state = STOPPED

    def clear(self) -> None:
        self.history = []
        self.rh = []

    def revoke(self) -> List[Tuple[str, str]]:
        if self.history and self.rh:
            self.history.pop()
            if cmd_opts.model is None or cmd_opts.model == "chatglm3":
                self.history.pop()
            self.rh.pop()
        return self.rh

    def limit_round(self):
        if cmd_opts.model is None or cmd_opts.model == "chatglm3":
            while len(self.history) >= self.max_rounds * 2:
                self.history.pop(0)
                self.history.pop(0)
                self.rh.pop(0)
        else:
            while len(self.history) >= self.max_rounds:
                self.history.pop(0)
                self.rh.pop(0)

    def limit_word(self):
        while self.get_word() > self.max_words:
            if cmd_opts.model is None or cmd_opts.model == "chatglm3":
                self.history.pop(0)
            self.history.pop(0)
            self.rh.pop(0)

    def get_round(self) -> int:
        if cmd_opts.model is None or cmd_opts.model == "chatglm3":
            return len(self.history) / 2
        return len(self.history)

    def get_word(self) -> int:
        prompt = ""
        if cmd_opts.model is None or cmd_opts.model == "chatglm3":
            for i, message in enumerate(self.history):
                prompt += "[Msg{}]{}".format(i, message['content'])
        else:
            for i, (old_query, response) in enumerate(self.history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        return len(prompt)

    def save_history(self):
        s = []
        if cmd_opts.model is None or cmd_opts.model == "chatglm3":
            s = self.history
        else:
            for i in self.history:
                s.append({'role': 'user', 'content': i[0]})
                s.append({'role': 'assistant', 'metadata': '', 'content': i[1]})
        filename = f"history-{int(time.time())}.json"
        p = os.path.join("outputs", "save", filename)
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(s, ensure_ascii=False))
        return f"成功保存至: {p}"

    def save_as_md(self):
        filename = f"history-{int(time.time())}.md"
        p = os.path.join("outputs", "markdown", filename)
        output = ""
        for i in self.history:
            if cmd_opts.model is None or cmd_opts.model == "chatglm3":
                output += f"# {i['role']}: {i['content']}\n\n"
            else:
                output += f"# 我: {i[0]}\n\nChatGLM: {i[1]}\n\n"
        with open(p, "w", encoding="utf-8") as f:
            f.write(output)
        return f"成功保存至: {p}"

    def load_history(self, file):
        try:
            with open(file.name, "r", encoding='utf-8') as f:
                j = json.load(f)
                _hist = []
                _readable_hist = []
                for i in j:
                    if i['role'] == 'user':
                        _readable_hist.append((i['content'], ''))
                    elif i['role'] == 'assistant':
                        _readable_hist[-1] = (_readable_hist[-1][0], i['content'])
                if cmd_opts.model is None or cmd_opts.model == "chatglm3":
                    _hist = j
                else:
                    _hist = _readable_hist.copy()
                _readable_hist = [(i[0], parse_codeblock(i[1])) for i in _readable_hist]
        except Exception as e:
            print('*' * 50)
            print(f"读取文件失败: {repr(e)}", end='')
            traceback.print_exception(e)
        self.history = _hist.copy()
        self.rh = _readable_hist.copy()
        return self.rh

    def edit_history(self, text, rnd_idx, obj_idx):
        if obj_idx == 0:
            if cmd_opts.model is None or cmd_opts.model == "chatglm3":
                self.history[rnd_idx * 2]['content'] = text
            else:
                self.history[rnd_idx] = (text, self.history[rnd_idx][1])
            self.rh[rnd_idx] = (text, self.rh[rnd_idx][1])
        elif obj_idx == 1:
            ok = parse_codeblock(text)
            if cmd_opts.model is None or cmd_opts.model == "chatglm3":
                self.history[rnd_idx * 2 + 1]['content'] = text
            else:
                self.history[rnd_idx] = (self.history[rnd_idx][0], text)
            self.rh[rnd_idx] = (self.rh[rnd_idx][0], ok)
        return self.rh

ctx = Context()
