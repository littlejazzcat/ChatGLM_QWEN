###########导入必要的库与模型初始设置（加载模型等）###############


from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import modelscope.models.audio.tts.sambert_hifi  #这个文件必须要先import，不然运行时可能出现解码错误
from datetime import datetime


import os
import json
import datetime
import platform
import torch

# 工具列表
#from langchain_community.tools.tavily_search import TavilySearchResults
import broadscope_bailian
from dashscope import Generation
import dashscope
dashscope.api_key="sk-HiBelOwiY2"


# tavily搜索引擎
from tavily import TavilyClient    # tavily搜索引擎api key
tavily = TavilyClient(api_key="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")

# 工具列表
from tool_list import tools,tool_info,prompt_tpl

tool_info = tool_info(tools)
tool_names,tool_descs = tool_info.tool_desc()


from qwen_agent_rewrite import llm,agent_execute,agent_execute_with_retry#导入qwen-agent
from tools import text2audio
from tool_list import system_item
from chatglm3_agent import agent
from tools import qwen_plus


import gradio as gr
import requests

chatglm3 = agent()


                
# 假设你的音频处理函数
def agent(user_query, history):
    history = history or []
    my_history=[system_item]
    result,my_history = chatglm3.llm_glm3(query = user_query,history=my_history)
    audio_dir = None
    if str(type(result)) == "<class 'dict'>":
        ###########执行对应工具，并将结果以 json 格式返回给模型并得到回复。
            if result['name'] == "text-to-speech":
                text = result['parameters']['text']
                audio_dir = text2audio(text)
                result = f'文字信息：{text}\n语音回复在下方语音框内'
                #with open(audio_dir, "r") as f:
                    #result = f.read()
            elif result['name'] == "qwen-plus":
                query = result['parameters']['text']
                answer = qwen_plus(query)
                result = f'正在进行搜索 \n 搜索结果：{answer}'
                
            elif result['name'] == "tavily":
                query = result['parameters']['text']
                answer = qwen_plus(query)
                result = f'正在进行搜索 \n 搜索结果：{answer}'
    
    history.append((user_query, result))
    return [history, audio_dir]


with gr.Blocks() as app:
    with gr.Row():
        chatbot = gr.Chatbot(label = 'ChatGLM-Qwen agent')
    with gr.Row():
        audio_box = gr.Audio()
    with gr.Row():
        query_box = gr.Textbox(label = '用户提问',autofocus = True,lines = 5)
    with gr.Row():
        #clear_btn = gr.ClearButton([query_box,chatbot,audio_box],value = '清空历史')
        submit_btn = gr.Button(value = '提交')

    def agent_chat(query,history):
        response = agent(query,history)
        print(response,type(response))
        yield '',response[0],response[1]
    
    #提交问题
    submit_btn.click(agent_chat,[query_box,chatbot],[query_box,chatbot,audio_box])
    query_box.submit(agent_chat,[query_box,chatbot],[query_box,chatbot,audio_box])

app.queue(200)
app.launch(max_threads = 500)
    