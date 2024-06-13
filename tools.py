##############该文件存放各种工具的执行函数



#############文本转语音工具：speech_sambert-hifigan_tts_zhitian_emo_zh-cn_16k
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from datetime import datetime

def text2audio(text = ''):
########## para:需要转换成语音的文本
    
    ######用当前时间作为音频文件的名称
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')
    audio_dir = f'/mnt/workspace/Agent/chatglm-qwen/audio/{formatted_datetime}.wav'

    model_id = 'damo/speech_sambert-hifigan_tts_zhitian_emo_zh-cn_16k'
    sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model=model_id)
    #print('success')
    output = sambert_hifigan_tts(input=text)
    wav = output[OutputKeys.OUTPUT_WAV]
    try:
        with open(audio_dir, 'wb') as f:
            f.write(wav)
        return audio_dir ########返回音频文件的dir
    except:
        print('写入文件失败')


############# qwen-agent调用 #############################################
import os
import json
from langchain_community.tools.tavily_search import TavilySearchResults
import broadscope_bailian
import datetime
from dashscope import Generation
import dashscope
dashscope.api_key="sk-HiBelOwiY2"
# travily搜索引擎
os.environ['TAVILY_API_KEY']='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'    # travily搜索引擎api key
tavily=TavilySearchResults(max_results=5)


# 工具列表
from tool_list import tools,tool_info,prompt_tpl

tool_info = tool_info(tools)
tool_names,tool_descs = tool_info.tool_desc()


from qwen_agent_rewrite import llm,agent_execute,agent_execute_with_retry#导入qwen-agent

def qwen_plus(query = '',role_content = ''):
    
    my_history=[]

    query = query
    #agent_execute_with_retry
    #paras:
    #      query : qwen-agent的输入，类型为字符串
    #      chat_history : 和qwen-agent的历史对话，作为chatglm3的协助agent可以直接给空列表
    #      retry_times : 默认为3次，可以自行调整从而修改qwen对于工具调用的精确度
    success,result,my_history = agent_execute_with_retry(query = query,chat_history=my_history,role_content = role_content)
    
    my_history=my_history[-10:]
    
    return result


################# tavily搜索工具#############
from tavily import TavilyClient

def tavily_search(query = ''):
    try:
        tavily = TavilyClient(api_key="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")
        # For basic search:
        response = tavily.search(query = query)
        # For advanced search:
        # response = tavily.search(query="Should I invest in Apple in 2024?", search_depth="advanced")
        # Get the search results as context to pass an LLM:
        context = [{"content": obj["content"]} for obj in response['results']]
        return context
        
    except:
        return
        