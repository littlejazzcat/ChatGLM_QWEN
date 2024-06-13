###########导入必要的库与模型初始设置（加载模型等）###############
import os
import json
import datetime
import platform
import torch
from transformers import AutoTokenizer, AutoModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from datetime import datetime
# 工具列表
from tools import text2audio
from tool_list import tools,system_item
from chatglm3_agent import agent



chatglm3 = agent()


while True:
    my_history=[system_item]
    query=input('query:')
    result,my_history = chatglm3.llm_glm3(query,history=my_history)
    ############当agent知道自己应该调用工具时会传回一个dict而不是str
    
    
    if str(type(result)) == "<class 'dict'>":
    ###########执行对应工具，并将结果以 json 格式返回给模型并得到回复。
        if result['name'] == "text-to-speech":
            text = str(result['parameters']['text'])
            audio_dir = text2audio(text)
            print(audio_dir)
        
        if result['name'] == "text-to-img":
            pass

