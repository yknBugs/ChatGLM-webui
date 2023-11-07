from typing import List, Tuple
import json
import os
import time
import traceback
from modules.options import cmd_opts

def parse_codeblock(text, is_disable):
    if is_disable:
        return str(text)
    if cmd_opts.model is None or cmd_opts.model == "chatglm3":
        text = str(text)
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
    
def parse_history(history, disable_parse):
    if cmd_opts.model is None or cmd_opts.model == "chatglm3":
        rh = []
        should_append = True
        for i in history:
            if i['role'] == 'user' or i['role'] == 'observation':
                rh.append((parse_codeblock(i['content'], disable_parse), None))
                should_append = False
            elif i['role'] == 'assistant':
                if should_append:
                    rh.append((None, parse_codeblock(i['content'], disable_parse)))
                else:
                    rh[-1] = (rh[-1][0], parse_codeblock(i['content'], disable_parse))
                    should_append = True
        return rh
    else:
        return [(i[0], parse_codeblock(i[1], disable_parse)) for i in history]

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
        self.sysprompt_value = 0
        self.disable_parse = False
        self.rh = []
        self.state = STOPPED
        self.max_rounds = 25
        self.max_words = 8192

    def sync_history(self):
        self.rh = parse_history(self.history, self.disable_parse)
        return self.rh

    def inferBegin(self):
        self.limit_round()
        self.limit_word()
        self.state = LOOP_FIRST

    def interrupt(self):
        if self.state == LOOP_FIRST or self.state == LOOP:
            self.state = INTERRUPTED

    def inferLoop(self, history) -> bool:
        # c: List[Tuple[str, str]]
        if self.state == INTERRUPTED:
            return True
        elif self.state == LOOP_FIRST:
            self.history = history
            self.rh = parse_history(self.history, self.disable_parse)
            self.state = LOOP
        else:
            self.history = history
            self.rh = parse_history(self.history, self.disable_parse)
        return False

    def inferEnd(self) -> None:
        if self.rh:
            self.rh = parse_history(self.history, self.disable_parse)
        self.state = STOPPED

    def clear(self) -> None:
        if self.sysprompt_value == 1:
            self.history = [self.history[0]]
        else:
            self.history = []
        self.rh = []

    def revoke(self) -> List[Tuple[str, str]]:
        if self.history and self.rh:
            self.history.pop()
            if cmd_opts.model is None or cmd_opts.model == "chatglm3":
                while self.history and self.history[-1]['role'] == 'assistant':
                    self.history.pop()
                self.history.pop()
            self.rh = parse_history(self.history, self.disable_parse)
        return self.rh

    def limit_round(self):
        if cmd_opts.model is None or cmd_opts.model == "chatglm3":
            while len(self.rh) >= self.max_rounds:
                self.history.pop(self.sysprompt_value)
                while len(self.history) > self.sysprompt_value and self.history[self.sysprompt_value]['role'] == 'assistant':
                    self.history.pop(self.sysprompt_value)
                self.rh = parse_history(self.history, self.disable_parse)
        else:
            while len(self.history) >= self.max_rounds:
                self.history.pop(self.sysprompt_value)
                self.rh = parse_history(self.history, self.disable_parse)

    def limit_word(self):
        while self.get_word() > self.max_words:
            self.history.pop(self.sysprompt_value)
            if cmd_opts.model is None or cmd_opts.model == "chatglm3":
                while len(self.history) > self.sysprompt_value and self.history[self.sysprompt_value]['role'] == 'assistant':
                    self.history.pop(self.sysprompt_value)
            self.rh = parse_history(self.history, self.disable_parse)

    def get_round(self) -> int:
        if cmd_opts.model is None or cmd_opts.model == "chatglm3":
            return len(self.rh)
        return len(self.history)

    def get_word(self) -> int:
        prompt = ""
        if cmd_opts.model is None or cmd_opts.model == "chatglm3":
            for i, message in enumerate(self.history):
                if message['role'] == 'system':
                    continue
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
        print('+' * 50)
        print(f"保存对话历史记录 json 文件至 {p}")
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
        print('+' * 50)
        print(f"保存对话历史记录 markdown 文件至 {p}")
        return f"成功保存至: {p}"

    def load_history(self, file):
        enable_sysprompt = False
        sysprompt = ''
        try:
            with open(file.name, "r", encoding='utf-8') as f:
                j = json.load(f)
                if len(j) > 0 and j[0]['role'] == 'system':
                    enable_sysprompt = True
                    sysprompt = j[0]['content']
                if cmd_opts.model is None or cmd_opts.model == "chatglm3":
                    _hist = j
                else:
                    # load ChatGLM3 history to ChatGLM2 and lower
                    enable_sysprompt = False
                    _hist = []
                    should_append = True
                    for i in j:
                        if i['role'] == 'user' or i['role'] == 'observation':
                            _hist.append((i['content'], " "))
                            should_append = False
                        elif i['role'] == 'assistant':
                            if should_append:
                                _hist.append((" ", i['content']))
                            else:
                                _hist[-1] = (_hist[-1][0], i['content'])
                                should_append = True
            self.history = _hist.copy()
            self.rh = parse_history(self.history, self.disable_parse)
            print('+' * 50)
            print("从文件读取历史记录")
            print(str(self.history))
        except Exception as e:
            print('*' * 50)
            print(f"读取文件失败: {repr(e)}")
            traceback.print_exc()
        if enable_sysprompt:
            return self.rh, True, {'value': sysprompt, 'visible': True, '__type__': 'update'}
        return self.rh, False, {'visible': False, '__type__': 'update'}
    
    def get_history(self, round_index, role_index):
        if cmd_opts.model is None or cmd_opts.model == "chatglm3":
            c = 0
            for i, message in enumerate(self.rh):
                if i == round_index:
                    break
                if message[0] is not None:
                    c += 1
                if message[1] is not None:
                    c += 1
            if role_index == 1 and self.rh[round_index][0] is not None:
                c += 1
            return c + self.sysprompt_value, 'content'
        else:
            return round_index, role_index

    def edit_history(self, text, round_index, role_index):
        if cmd_opts.model is None or cmd_opts.model == "chatglm3":
            self.history[round_index][role_index] = text
        else:
            if role_index == 0:
                self.history[round_index] = (text, self.history[round_index][1])
            elif role_index == 1:
                self.history[round_index] = (self.history[round_index][0], text)
        self.rh = parse_history(self.history, self.disable_parse)
        return self.rh

ctx = Context()
