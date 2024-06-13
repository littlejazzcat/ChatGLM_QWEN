"""这个文件用于定义agent需要使用到的工具"""

tools = [
    {
        "name": "tavily",
        "description": "这是一个类似谷歌和百度的搜索引擎，搜索知识、天气、股票、电影、小说、百科等都是支持的，如果你不确定就搜索一下",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "description": "需要搜索的内容"
                }
            },
            "required": ['text']
        }
    },
    {
        "name": "text-to-speech", #####文本转语音模型使用speech_sambert-hifigan_tts_zhitian_emo_zh-cn_16k
        "description": "将文本转换为语音",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "description": "需要转换成语音的文本"
                },
            },
            "required": ['text']
        }
    },
    {#,搜索知识、天气、股票、电影、小说、百科等都是支持的
        "name": "qwen-plus",
        "description": "这是另一个人工智能体，如果你不确定可以向qwen提问",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "description": "需要提问的内容"
                },
            },
            "required": ['text']
        }
    },

]
prompt = {
    "role": "system",
    "content": "Answer the following questions as best as you can. You have access to the following tools:",
    "tools": tools
}


# 工具列表
# tools=[]
import os
import json
class tool_info:
    def __init__(self,tools):
        self.tools = tools
    
    def tool_desc(self):
        tool_names=' or '.join([tool['name'] for tool in self.tools])  # 拼接工具名
        tool_descs=[] # 拼接工具详情
        for t in self.tools:
            args_desc=[]
            for name,info in t['parameters']['properties'].items():
                args_desc.append({'name':name,'description':info['description'] if 'description' in info else '','type':t['parameters']['type']})
            #args_desc=json.dumps(args_desc,ensure_ascii=False)   #这个是需要传给工具的参数，需要转换成json格式
            tool_descs.append('%s: %s,args: %s'%(t['name'],t['description'],args_desc))
        tool_descs='\n'.join(tool_descs)
        return tool_names,tool_descs

##############这个prompt适用于qwen模型############
prompt_tpl='''Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

These are chat history before:
{chat_history}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
{agent_scratchpad}
'''

###############用于chatglm3的prompt###############
system_item = {
    "role": "system",
    "content": "Answer the following questions as best as you can. You have access to the following tools:",
    "tools": tools
}
