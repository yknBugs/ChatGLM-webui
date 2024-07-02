import os

import gradio as gr

from modules import options
from modules.context import Context
from modules.model import infer

css = "style.css"
script_path = "scripts"
_gradio_template_response_orig = gr.routes.templates.TemplateResponse

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

def predict(ctx, query, max_length, top_p, temperature, use_stream_chat):
    ctx.inferBegin()
    token = 0
    ctx_round = ctx.get_round()
    ctx_word = ctx.get_word()
    yield ctx.rh, "正在生成回复内容...", f"总对话轮数: {ctx_round}\n总对话字数: {ctx_word}\nToken 数: {token}"

    for _, output, history in infer(
            query=query,
            history=ctx.history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            use_stream_chat=use_stream_chat
    ):
        if ctx.inferLoop(history):
            print("")
            break

        token += 1
        yield ctx.rh, gr_show(), f"总对话轮数: {ctx_round}\n总对话字数: {ctx_word}\nToken 数: {token}"

    ctx.inferEnd()
    yield ctx.rh, "", f"总对话轮数: {ctx.get_round()}\n总对话字数: {ctx.get_word()}\n上次回复 Token 数: {token}"

def regenerate(ctx, max_length, top_p, temperature, use_stream_chat):
    if not ctx.rh:
        print('*' * 50)
        raise RuntimeError("没有过去的对话")
    
    i = ctx.history.pop()
    if options.cmd_opts.model is None or options.cmd_opts.model == "chatglm3" or options.cmd_opts.model == "chatglm4":
        while ctx.history and ctx.history[-1]['role'] == 'assistant':
            i = ctx.history.pop()
        i = ctx.history.pop()
        query = i['content']
    else:
        query = i[0]
    ctx.sync_history()

    print('+' * 50)
    print("撤回上一条消息")

    for p0, p1, p2 in predict(ctx, query, max_length, top_p, temperature, use_stream_chat):
        yield p0, p1, p2

def revoke(ctx):
    print('+' * 50)
    print("撤回上一条消息")
    return ctx.revoke(), "已撤回上一条消息", {'visible': False, '__type__': 'update'}, {'value': '', 'label': '', '__type__': 'update'}, []

def clear_history(ctx):
    ctx.clear()
    print('+' * 50)
    print("清空对话")
    return gr.update(value=[]), "已清空对话", {'visible': False, '__type__': 'update'}, {'value': '', 'label': '', '__type__': 'update'}, []

def edit_history(ctx, log, idx):
    round_index, role_index = ctx.get_history(idx[0], idx[1])
    if log == '':
        return ctx.rh, {'visible': True, '__type__': 'update'},  {'value': ctx.history[round_index][role_index], '__type__': 'update'}, idx
    print('+' * 50)
    print(ctx.history[round_index][role_index])
    print("----->")
    print(log)
    ctx.edit_history(log, round_index, role_index)
    return ctx.rh, *gr_hide()

def gr_show_and_load(ctx, evt: gr.SelectData):
    if evt.index[1] == 0:
        label = f'修改提问内容{evt.index[0]}：'
    else:
        label = f'修改回答内容{evt.index[0]}：'
    round_index, role_index = ctx.get_history(evt.index[0], evt.index[1])
    return {'visible': True, '__type__': 'update'}, {'value': ctx.history[round_index][role_index], 'label': label, '__type__': 'update'}, evt.index

def gr_hide():
    return {'visible': False, '__type__': 'update'}, {'value': '', 'label': '', '__type__': 'update'}, []

def apply_max_round_click(ctx, max_round):
    print('+' * 50)
    print(f"设置最大对话轮数 {ctx.max_rounds} -> {max_round}")
    ctx.max_rounds = max_round
    return f"成功设置: 最大对话轮数 {ctx.max_rounds}"

def apply_max_words_click(ctx, max_words):
    print('+' * 50)
    print(f"设置最大对话字数 {ctx.max_words} -> {max_words}")
    ctx.max_words = max_words
    return f"成功设置: 最大对话字数 {ctx.max_words}" 

def enable_system_prompt(ctx, value, prompt):
    if options.cmd_opts.model is None or options.cmd_opts.model == "chatglm3" or options.cmd_opts.model == "chatglm4":
        if value == True:
            ctx.sysprompt_value = 1
            if len(ctx.history) == 0 or ctx.history[0]['role'] != 'system':
                ctx.history.insert(0, {'role': 'system', 'content': prompt})
            else:
                ctx.history[0]['content'] = prompt
            print('+' * 50)
            print(f'System Prompt: {prompt}')
            return "系统(全局)提示词已启用", gr_show(), gr_show(), gr_show()
        else:
            ctx.sysprompt_value = 0
            new_prompt = prompt
            if len(ctx.history) > 0 and ctx.history[0]['role'] == 'system':
                new_prompt = ctx.history[0]['content']
                ctx.history.pop(0)
            print('+' * 50)
            print(f'禁用 System Prompt: {new_prompt}')
            return "系统(全局)提示词已禁用", {'value': new_prompt, 'visible': False, '__type__': 'update'}, gr_show(False), gr_show(False)
    return "此模型暂不支持使用系统(全局)提示词", gr_show(False), gr_show(False), gr_show(False)

def submit_system_prompt(ctx, prompt):
    if options.cmd_opts.model is None or options.cmd_opts.model == "chatglm3" or options.cmd_opts.model == "chatglm4":
        ctx.sysprompt_value = 1
        if len(ctx.history) == 0 or ctx.history[0]['role'] != 'system':
            ctx.history.insert(0, {'role': 'system', 'content': prompt})
        else:
            ctx.history[0]['content'] = prompt
        print('+' * 50)
        print(f'System Prompt: {prompt}')
        return "系统(全局)提示词已更新"
    return "此模型暂不支持使用系统(全局)提示词"

def undo_system_prompt(ctx):
    if options.cmd_opts.model is None or options.cmd_opts.model == "chatglm3" or options.cmd_opts.model == "chatglm4":
        if len(ctx.history) == 0 or ctx.history[0]['role'] != 'system':
            return "你是ChatGLM，由智谱AI训练的一个语言模型，请根据用户的指示正确的回答用户的问题。", "已恢复默认的系统(全局)提示词"
        else:
            return ctx.history[0]['content'], "已撤销对系统(全局)提示词的更改"
    return "你是ChatGLM，由智谱AI训练的一个语言模型，请根据用户的指示正确的回答用户的问题。", "此模型暂不支持使用系统(全局)提示词"

def create_ui():
    reload_javascript()

    with gr.Blocks(css=css, analytics_enabled=False) as chat_interface:
        _ctx = Context()
        state = gr.State(_ctx)
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""<h2><center>ChatGLM WebUI</center></h2>""")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            max_length = gr.Slider(minimum=64, maximum=1048576, step=64, label='Max Length', value=16384)
                            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=0.8)
                        with gr.Row():
                            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature', value=0.95)

                        with gr.Row():
                            max_rounds = gr.Slider(minimum=1, maximum=100, step=1, label="最大对话轮数", value=25)
                            apply_max_rounds = gr.Button("✔", elem_id="del-btn")

                        with gr.Row():
                            max_words = gr.Slider(minimum=64, maximum=8388608, step=64, label='最大对话字数', value=32768)
                            apply_max_words = gr.Button("✔", elem_id="del-btn")

                        cmd_output = gr.Textbox(label="消息输出", interactive=False)
                        with gr.Row():
                            use_stream_chat = gr.Checkbox(label='使用流式输出', value=True)
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear_history_btn = gr.Button("清空对话")

                        with gr.Row():
                            sync_his_btn = gr.Button("同步对话")

                        with gr.Row():
                            save_his_btn = gr.Button("保存对话")
                            load_his_btn = gr.UploadButton("读取对话", file_types=['file'], file_count='single')

                        with gr.Row():
                            save_md_btn = gr.Button("保存为 MarkDown")

                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            gr.Markdown('''说明:<br/>`Max Length` 生成文本时的长度限制<br/>`Top P` 控制输出文本中概率最高前 p 个单词的总概率<br/>`Temperature` 控制生成文本的多样性和随机性<br/>`Top P` 变小会生成更多样和不相关的文本；变大会生成更保守和相关的文本。<br/>`Temperature` 变小会生成更保守和相关的文本；变大会生成更奇特和不相关的文本。<br/>`最大对话轮数` 对话记忆轮数<br/>`最大对话字数` 对话记忆字数<br/>限制记忆可减小显存占用。<br/>点击对话可直接修改对话内容''')

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False, height=800)
                with gr.Row(visible=False) as edit_log:
                    with gr.Column():
                        log = gr.Textbox(placeholder="输入你修改后的内容", lines=4, elem_id="chat-input", container=False)
                        with gr.Row():
                            submit_log = gr.Button('保存', variant="primary")
                            cancel_log = gr.Button('取消')
                log_idx = gr.State([])

                with gr.Row():
                    input_message = gr.Textbox(placeholder="输入你的内容...(按 Ctrl+Enter 发送)", show_label=False, lines=4, elem_id="chat-input", container=False)
                    clear_input = gr.Button("🗑️", elem_id="del-btn")
                    stop_generate = gr.Button("❌", elem_id="del-btn")

                with gr.Row():
                    submit = gr.Button("发送", elem_id="c_generate", variant="primary")

                with gr.Row():
                    revoke_btn = gr.Button("撤回")
                
                with gr.Row():
                    regenerate_btn = gr.Button("重新生成")

        submit.click(predict, inputs=[
            state,
            input_message,
            max_length,
            top_p,
            temperature,
            use_stream_chat
        ], outputs=[
            chatbot,
            input_message,
            cmd_output
        ])

        regenerate_btn.click(regenerate, inputs=[
            state,
            max_length,
            top_p,
            temperature,
            use_stream_chat
        ], outputs=[
            chatbot,
            input_message,
            cmd_output
        ])
        
        revoke_btn.click(revoke, inputs=[state], outputs=[chatbot, cmd_output, edit_log, log, log_idx])
        clear_history_btn.click(clear_history, inputs=[state], outputs=[chatbot, cmd_output, edit_log, log, log_idx])
        stop_generate.click(lambda ctx: ctx.interrupt(), inputs=[state], outputs=[])
        clear_input.click(lambda x: "", inputs=[input_message], outputs=[input_message])
        save_his_btn.click(lambda ctx: ctx.save_history(), inputs=[state], outputs=[cmd_output])
        save_md_btn.click(lambda ctx: ctx.save_as_md(), inputs=[state], outputs=[cmd_output])
        sync_his_btn.click(lambda ctx: ctx.rh, inputs=[state], outputs=[chatbot])
        apply_max_rounds.click(apply_max_round_click, inputs=[state, max_rounds], outputs=[cmd_output])
        apply_max_words.click(apply_max_words_click, inputs=[state, max_words], outputs=[cmd_output])
        chatbot.select(gr_show_and_load, inputs=[state], outputs=[edit_log, log, log_idx])
        submit_log.click(edit_history, inputs=[state, log, log_idx], outputs=[chatbot, edit_log, log, log_idx])
        cancel_log.click(gr_hide, outputs=[edit_log, log, log_idx])

    with gr.Blocks(css=css, analytics_enabled=False) as settings_interface:
        with gr.Row():
            setting_output = gr.Textbox(label="消息输出", interactive=False)

        with gr.Row():
            disable_parse = gr.Checkbox(True, label="禁用字符转义")

        with gr.Row():
            enable_sysprompt = gr.Checkbox(False, label="启动系统(全局)提示词")
        
        with gr.Row():
            system_prompt = gr.Textbox(value="你是ChatGLM，由智谱AI训练的一个语言模型，请根据用户的指示正确的回答用户的问题。", placeholder="输入系统(全局)提示词的内容", visible=False, show_label=False, lines=10, container=False)
        
        with gr.Row():
            submit_sysprompt = gr.Button("更新系统(全局)提示词", variant="primary", visible=False)
            undo_sysprompt = gr.Button("放弃系统(全局)提示词更改", visible=False)

        with gr.Row():
            reload_ui = gr.Button("重启 WebUI")

        def disable_text_parse(ctx, value):
            if value:
                ctx.disable_parse = True
                return "已禁用字符转义", ctx.sync_history()
            ctx.disable_parse = False
            return "已启用字符转义", ctx.sync_history()

        def restart_ui():
            options.need_restart = True

        enable_sysprompt.change(enable_system_prompt, inputs=[state, enable_sysprompt, system_prompt], outputs=[setting_output, system_prompt, submit_sysprompt, undo_sysprompt])
        submit_sysprompt.click(submit_system_prompt, inputs=[state, system_prompt], outputs=[setting_output])
        undo_sysprompt.click(undo_system_prompt, inputs=[state], outputs=[system_prompt, setting_output])
        load_his_btn.upload(lambda ctx, f: ctx.load_history(f), inputs=[state, load_his_btn], outputs=[chatbot, enable_sysprompt, system_prompt])
        disable_parse.change(disable_text_parse, inputs=[state, disable_parse], outputs=[setting_output, chatbot])
        reload_ui.click(restart_ui)

    interfaces = [
        (chat_interface, "聊天", "chat"),
        (settings_interface, "设置", "settings")
    ]

    with gr.Blocks(css=css, analytics_enabled=False, title="ChatGLM") as demo:
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id="tab_" + ifid):
                    interface.render()

    return demo


def reload_javascript():
    scripts_list = [os.path.join(script_path, i) for i in os.listdir(script_path) if i.endswith(".js")]
    javascript = ""
    # with open("script.js", "r", encoding="utf8") as js_file:
    #     javascript = f'<script>{js_file.read()}</script>'

    for path in scripts_list:
        with open(path, "r", encoding="utf8") as js_file:
            javascript += f"\n<script>{js_file.read()}</script>"

    # todo: theme
    # if cmd_opts.theme is not None:
    #     javascript += f"\n<script>set_theme('{cmd_opts.theme}');</script>\n"

    def template_response(*args, **kwargs):
        res = _gradio_template_response_orig(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
