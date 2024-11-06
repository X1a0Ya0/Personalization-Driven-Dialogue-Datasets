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
    "1. Do you like this story?",
    "2. Please summarize the main content of the story.",
    "3. What did you learn from the story?",
    "4. Do you agree with this opinion,  Tell me what you think?",
    "5. When you read this story, do you think of your own experience?"
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
chat_glmagent= ConversableAgent(name="chat_glm",llm_config=gpt,system_message="Your task is to evaluate the conversation",human_input_mode="NEVER")

def generate_prompt(conversation):
    prompt = f"Based on the {conversation}, please judge the fluency and coherence of the answer separately. You only need to answer whether it is readable and relevant. Each conversation follows the same answer format. Fluency: measures whether the answer is readable. The answer should be a sentence that humans can understand; Coherence: evaluates the relevance of the answer to the question. The answer should be consistent with the topic of the question, not irrelevant;"
    return prompt
for conversation in final_conversations:
    prompt = generate_prompt(conversation)
    prompt = [{"role": "user", "content": f"{prompt}"}]
    response = chat_glmagent.generate_reply(messages=prompt, sender=chat_glmagent)
    print(response)
