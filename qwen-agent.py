"""重写qwen_agent的工具调用prompt"""
import os
import json
from langchain_community.tools.tavily_search import TavilySearchResults
import broadscope_bailian
import datetime
from dashscope import Generation
import dashscope
dashscope.api_key="sk-HiBelOwiY2"

def llm(query,history=[],user_stop_words=[],role_content = ''):    # 调用api_server

    try:
        if role_content:
            messages=[{'role':'system','content':f'{role_content}'}]
        else:
            messages=[{'role':'system','content':'You are a helpful assistant.'}]
        #####将历史对话加入下一次传入的messages中
        if history:
            for hist in history:
                messages.append({'role':'user','content':hist[0]})
                messages.append({'role':'assistant','content':hist[1]})
        messages.append({'role':'user','content':query})

        resp = Generation.call(
            Generation.Models.qwen_plus,
            messages=messages,
            result_format='message',  # 设置结果为 "message" 格式.
        )
        print(type(resp))
        #print(resp)
        content=resp.get("output", {}).get("choices", [])[0].get("message", {}).get("content")
        return content
    except Exception as e:
        return str(e)
    
# travily搜索引擎
from tavily import TavilyClient
#os.environ['TAVILY_API_KEY']='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'    # travily搜索引擎api key
#'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'
tavily = TavilyClient(api_key="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")


# 工具列表
from tool_list import tools,tool_info,prompt_tpl

tool_info = tool_info(tools)
tool_names,tool_descs = tool_info.tool_desc()



"""
para:
        query:用户提问内容
        chat_history:历史对话数据
"""
def agent_execute(query,chat_history = [] ,role_content = ''):
    global tools,tool_names,tool_descs,prompt_tpl,llm,tokenizer
    role_content = role_content
    history = [] #初始化历史记录
    agent_scratchpad = '' # agent执行过程
    while True:
        # 1）触发llm思考下一步action
        if chat_history:
            history = '\n'.join(['Question:%s\nAnswer:%s'%(his[0],his[1]) for his in chat_history])
        
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        prompt = prompt_tpl.format(today=today,chat_history=history,tool_descs=tool_descs,tool_names=tool_names,query=query,agent_scratchpad=agent_scratchpad)
        #print('\033[32m---等待LLM返回... ...\n%s\n\033[0m'%prompt,flush=True)
        response = llm(prompt,user_stop_words=['Observation:'],role_content = role_content)
        print('\033[34m---LLM返回---\n%s\n---\033[34m'%response,flush=True)
        
        # 2）解析thought+action+action input+observation or thought+final answer
        thought_i = response.rfind('Thought:')
        final_answer_i = response.rfind('\nFinal Answer:')
        action_i = response.rfind('\nAction:')
        action_input_i = response.rfind('\nAction Input:')
        observation_i = response.rfind('\nObservation:')
        
        # 3）返回final answer，执行完成
        if final_answer_i != -1 and thought_i<final_answer_i:
            final_answer = response[final_answer_i+len('\nFinal Answer:'):].strip()
            chat_history.append((query,final_answer))
            return True,final_answer,chat_history
        
        # 4）解析action
        if not (thought_i<action_i<action_input_i):
            return False,'LLM回复格式异常',chat_history
        if observation_i == -1:
            observation_i = len(response)
            response = response+'Observation: '
        thought = response[thought_i+len('Thought:'):action_i].strip()
        action = response[action_i+len('\nAction:'):action_input_i].strip()
        action_input = response[action_input_i+len('\nAction Input:'):observation_i].strip()
        #print(f'test:{action_input}')
        # 5）匹配tool
        the_tool=None
        for t in tools:
            if t['name'] == action:
                the_tool = t
                break
        if the_tool is None:
            observation = 'the tool not exist'
            agent_scratchpad = agent_scratchpad+response+observation+'\n'
            continue 
        
        # 6）执行tool
        try:
            #action_input = json.loads(action_input)
            ######这里直接确定只用tavily了，其他工具调用需要另写一个
            tool_ret = tavily.search(query = str(action_input))
            print(tool_ret)
        except Exception as e:
            observation = 'the tool has error:{}'.format(e)
        else:
            observation = str(tool_ret)
        agent_scratchpad = agent_scratchpad+response+observation+'\n'

        
        
############ agent多轮思考调用，可以根据设置retry_times参数修改思考轮数########

def agent_execute_with_retry(query,chat_history=[],retry_times=3, role_content = ''):
    for i in range(retry_times):
        role_content = role_content
        success,result,chat_history=agent_execute(query,chat_history=chat_history,role_content = role_content)
        if success:
            return success,result,chat_history
    return success,result,chat_history


if __name__ == '__main__':
    my_history=[]
    while True:
        query=input('query:')
        success,result,my_history=agent_execute_with_retry(query,chat_history=my_history)
        my_history=my_history[-10:]
