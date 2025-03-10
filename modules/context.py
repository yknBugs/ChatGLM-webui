from typing import Optional, List, Tuple
import json
import os
import time


def parse_codeblock(text):
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
    def __init__(self, history: Optional[List[Tuple[str, str]]] = None):
        if history:
            self.history = history
        else:
            self.history = []
        self.rh = []
        self.state = STOPPED
        self.max_rounds = 25
        self.max_words = 8192

    def inferBegin(self):
        self.state = LOOP_FIRST

        hl = len(self.history)
        if hl == 0:
            return
        elif hl == self.max_rounds:
            self.history.pop(0)
            self.rh.pop(0)
        elif hl > self.max_rounds:
            self.history = self.history[-self.max_rounds:]
            self.rh = self.rh[-self.max_rounds:]

    def interrupt(self):
        if self.state == LOOP_FIRST or self.state == LOOP:
            self.state = INTERRUPTED

    def inferLoop(self, query, output) -> bool:
        # c: List[Tuple[str, str]]
        if self.state == INTERRUPTED:
            return True
        elif self.state == LOOP_FIRST:
            self.history.append((query, output))
            self.rh.append((query, parse_codeblock(output)))
            self.state = LOOP
        else:
            self.history[-1] = (query, output)
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
            self.rh.pop()
        return self.rh

    def limit_round(self):
        hl = len(self.history)
        if hl == 0:
            return
        elif hl == self.max_rounds:
            self.history.pop(0)
            self.rh.pop(0)
        elif hl > self.max_rounds:
            self.history = self.history[-self.max_rounds:]
            self.rh = self.rh[-self.max_rounds:]

    def limit_word(self):
        prompt = ""
        for i, (old_query, response) in enumerate(self.history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        while len(prompt) > self.max_words:
            self.history.pop(0)
            self.rh.pop(0)

            prompt = ""
            for i, (old_query, response) in enumerate(self.history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)

    def get_round(self) -> int:
        return len(self.history)

    def get_word(self) -> int:
        prompt = ""
        for i, (old_query, response) in enumerate(self.history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        return len(prompt)

    def save_history(self):
        s = [{"q": i[0], "o": i[1]} for i in self.history]
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
            output += f"# 我: {i[0]}\n\nChatGLM: {i[1]}\n\n"
        with open(p, "w", encoding="utf-8") as f:
            f.write(output)
        return f"成功保存至: {p}"

    def load_history(self, file):
        try:
            with open(file.name, "r", encoding='utf-8') as f:
                j = json.load(f)
                _hist = [(i["q"], i["o"]) for i in j]
                _readable_hist = [(i["q"], parse_codeblock(i["o"])) for i in j]
        except Exception as e:
            print(e)
        self.history = _hist.copy()
        self.rh = _readable_hist.copy()
        return self.rh

    def edit_history(self, text, rnd_idx, obj_idx):
        if obj_idx == 0:
            self.history[rnd_idx] = (text, self.history[rnd_idx][1])
            self.rh[rnd_idx] = (text, self.rh[rnd_idx][1])
        elif obj_idx == 1:
            ok = parse_codeblock(text)
            self.history[rnd_idx] = (self.history[rnd_idx][0], text)
            self.rh[rnd_idx] = (self.rh[rnd_idx][0], ok)
        return self.rh

ctx = Context()
