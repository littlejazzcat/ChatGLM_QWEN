"""这个文件用于设计基于chatglm3的agent"""


###########导入必要的库与模型初始设置（加载模型等）###############
import os
import json
import datetime
import platform
import torch
from transformers import AutoTokenizer, AutoModel

# 工具列表
from tool_list import tools,system_item


class agent():
    def __init__(self):
        self.MODEL_PATH = os.environ.get('MODEL_PATH', '/mnt/workspace/ChatGLM3/chatglm3-6b')
        self.TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", self.MODEL_PATH)

        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_PATH, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

        self.os_name = platform.system()
        self.clear_command = 'cls' if self.os_name == 'Windows' else 'clear'
        self.stop_stream = False

    ######### chatglm3模型调用 ###########
    def llm_glm3(self,query,history=[]):
        past_key_values, history = None, history
        role = "user"

        #########将参数传入模型并收集输出结果###########
        #### chat方法参数：query必须是字符串类型
        response, history = self.model.chat(self.tokenizer, query, history=history, role = role)

        if str(type(response)) != "<class 'dict'>":
            return response,history
        else:
            print('answer: 正在调用工具中，请稍等')

        return response,history
    
    #########chatglm3自定义角色调用#######

    
    
    def llm_glm3_observation(self,result,history):
        response, history = self.model.chat(self.tokenizer, result, history=history, role="observation")
        
        return response, history


    def chatglm3_agent_run(self):
        my_history=[system_item]
        while True:
            query=input('query:')
            result,my_history = llm_glm3(query,chat_history=my_history)
            print('\033[34m---LLM返回---\n%s\n---\033[34m'%result,flush=True)
            print(type(result))
            #my_history=my_history[-10:]