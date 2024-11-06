#-*- coding:utf-8 -*-
import pandas as pd
from autogen import ConversableAgent

import re
qwen2_7b =  {
    "config_list": [
    {
        "model":"Qwen/Qwen2.5-72B-Instruct",
        "base_url":"https://api.siliconflow.cn/v1",
        "api_key":"your_api_key"
    },
    ],
    "cache_seed": None,
}
gpt =  {
    "config_list": [
    {
        "model":"gpt-4o",
        "api_key":"your_api_key"
    },
    ],
    "cache_seed": None,
}

df = pd.read_csv('output.csv',encoding='gbk')
conversation_column = df['text'].dropna()
conversation_list = [' '.join(conversation_column[i:i+5]) for i in range(0, len(conversation_column), 5)]
questions = [
    "1. 你喜欢这个故事吗？",
    "2. 请概括一下故事的主要内容。",
    "3. 从故事中你学到了什么？",
    "4. 你赞同这个道理吗？说说你的看法。",
    "5. 读了这段故事，是否联想到了你自身的经历？"
]

# 定义正则表达式来提取以序号开头的对话
pattern = r'\d+\.\s*(.*?)(?=\d+\.|$)'

# 最终结果的列表
final_conversations = []

# 遍历每个对话组
for conversation in conversation_list:
    # 使用正则表达式提取对话中的段落
    split_conversation = re.findall(pattern, conversation)

    # 将问题与提取的对话内容结合，避免索引超出范围
    combined_conversation = []
    for i in range(len(questions)):
        if i < len(split_conversation):  # 确保不会超出对话的句子数量
            combined_conversation.append(f"{questions[i]} {split_conversation[i].strip()}")

    # 将组合好的对话存入最终列表
    final_conversations.append(combined_conversation)
score_results = []
chat_glmagent= ConversableAgent(name="chat_glm",llm_config=qwen2_7b,system_message="你的任务是按照要求做事",human_input_mode="NEVER")
question=""
def generate_prompt(conversation):
    prompt = f"根据{conversation}的内容，请分别判断回答的流利性和连贯性，你只需要回答是否可读和是否相关。每段对话按同一种回答格式。流利性：衡量回答是否可读，回答应该是人类可以理解的句子;连贯性：评估回答和问题的相关性，回答应该与问题的主题一致，而不是与主题无关;"
    return prompt
for conversation in final_conversations:
    prompt = generate_prompt(conversation)
    prompt = [{"role": "user", "content": f"{prompt}"}]
    response = chat_glmagent.generate_reply(messages=prompt, sender=chat_glmagent)
    print(response)
