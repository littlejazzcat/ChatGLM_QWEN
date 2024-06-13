###########导入必要的库与模型初始设置（加载模型等）###############

import modelscope.models.audio.tts.sambert_hifi  #这个文件必须要先import，不然运行时可能出现解码错误

from IPython.display import Audio
import numpy as np
import time

# 工具列表
import tools
from tools import text2audio
from tools import qwen_plus
from tool_list import tools,system_item
from chatglm3_agent import agent

#tavily搜索引擎
from tavily import TavilyClient    # tavily搜索引擎api key
tavily = TavilyClient(api_key="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")

#################实例化agent对象#############################
chatglm3 = agent()

#########################################################

##########################工具调用demo#################
while True:
    my_history=[system_item]
    query=input('query:')
    result,my_history = chatglm3.llm_glm3(query,history=my_history)
    if str(type(result)) == "<class 'dict'>":
        ###########执行对应工具，并将结果以 json 格式返回给模型并得到回复。
            if result['name'] == "text-to-speech":
                text = result['parameters']['text']
                audio_dir = text2audio(text)
                print('answer: 语音回复如下')
                audio_path = audio_dir
                # 播放WAV文件
                Audio(audio_path)
            elif result['name'] == "tavily":
                query = result['parameters']['text']
                answer = qwen_plus(query)
                #print(answer)
                #result = json.dumps({"search": 12412}, ensure_ascii=False)
                #response, history = chatglm3.llm_glm3_observation(answer,history = my_history)
                result = answer
                print('answer：',result)
    else:
        print(result)

#######################################################################

##########################  agent对话demo  #################
while True:
    
    try:
        chat_times = input('请输入对话轮数：')
        chat_times = int(chat_times)
        if chat_times == 0:
            break
    except:
        print('请输入一个正整数')
        continue
    chatglm3_role = input('请输入chatglm3扮演的角色：')
    qwen_role = input('请输入qwen-agent扮演的角色：')
    query = input('query:')
    
    system_item['content'] = chatglm3_role
    my_history=[system_item]
    
    while chat_times > 0:
        chat_times -= 1
        
        result,my_history = chatglm3.llm_glm3(query,history=my_history)
        query = result
        print(f'chatglm3：{query}')
        ########将chatglm3的返回结果传给qwen作为query
        answer = qwen_plus(query,role_content = qwen_role)
        print(f'qwen_plus：{answer}')
        #my_history.append([query, answer])
        query = answer
        time.sleep(1)
        
    