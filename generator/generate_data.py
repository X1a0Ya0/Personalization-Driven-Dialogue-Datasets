#-*- coding:utf-8 -*-
import json
import csv
from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.teachability import Teachability

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
with open('/Ch_stories.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]
story = data[0]['story']
with open("./En-profile.json", 'r', encoding='utf-8') as file:
    agent_attributes = json.load(file)
agent_attributes_str = json.dumps(agent_attributes)
question = "1.你喜欢这个故事吗？2.请概括一下故事的主要内容 3.从故事中你学到了什么？4.你赞同这个道理吗，说说你的看法？5.读了这段故事，是否联想到了你自身的经历？"
description = "其中大五人格特性的定义如下：Openness{[high]: 喜欢这个故事;[low]:不喜欢这个故事}\nConscientiousness{[high]: 对故事概括十分全面;[low]: 忽略故事的情节}\nExtraversion：{[high]:回答态度热情;[low]回答态度冷漠: }\nAgreeableness：{[high]:在交流过程多用'请'，'您'等文明用语;[low]: 不喜欢回答问题}\nNeuroticism：{[high]: 回答时出现用省略号表示的停顿;[low]:回答通顺}"
teachable_agent = ConversableAgent(
    name="teachable_agent",
    system_message=f"1. 阅读{story}模拟给定信息的角色，根据基本信息{agent_attributes_str}，回答{question}",# The name is flexible, but should not contain spaces to work in group chat.
    llm_config=qwen2_7b  # Disable caching.
)

# teachability = Teachability(
#     verbosity=1,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
#     reset_db=True,
#     path_to_db_dir="./tmp/notebook/teachability_db",
#     recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
# )
# teachability.add_to_agent(teachable_agent)
user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=0,
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

example =[
       { "trait": "{'gender': 'male', 'age': 25, 'vocation': 'student', 'education_level': 'bachelor', 'big_five_personality': {'Openness': 'low', 'Conscientiousness': 'high', 'Extraversion': 'high', 'Agreeableness': 'high', 'Neuroticism': 'low'}, 'emotional': 'relax', 'sentiment': 'neutral'}",
          "thought":"充分考虑性格属性回答问题，因为我的openness属性为low，所以我对于故事的内容并不感兴趣;由于我的conscientiousness属性为high，所以我能够很全面的概括故事细节;我的Extraversion属性为high，因此我能够很好的配合回答;此外，高的Agreeableness使我在回答问题时，能表现出礼貌。最后，较低的Neuroticism使我的情绪较为稳定，回答的字数较多。结合我的emotional以及sentiment的状态，我的回答如下",
          "answer":"1.不好意思，这个故事并没有很好的打动到我，所以对于这个故事我并不喜欢。2. 这个故事主要讲述了一个苏格兰人去伦敦，想要联系一位老朋友。但是不清楚朋友的联系方式，向父亲'求救’,但是父亲却会错了意，从而引发的笑话。3. 从这个故事中，我学习到了有时候最重要的往往是最没有意义的。就像苏格兰人得到了关于父亲的答案，但是并没有帮他解决自身的问题。4. 我非常赞同这个观点！这个观点给予了我非常大的启示，会让我停下来思考一下自己目前所追求的是否是真正有意义的事情。真是太棒了!5.读了这个故事，回顾我的读书生涯，我发现我好像一直在追求成绩的好坏，分数的高低。以为这是能实现我人生意义的事情。但是渐渐的我发现分数的高低并不代表我的价值有多少。注意哦，这里并不是说分数完全没有用，而是说真正有意义的是学会那些技能和知识，就像故事中真正有用的是托马的住址而非父亲的知道与否。所以，希望大家额能够找到真正有意义的事情，而不是被其他的干扰所牵绊!"},
]
def prompt(trait,example):
    prompts= f"根据{trait}的属性，请直给出问题的答案。符合角色大五人格的文本可以参考一下例子，例如：{example}。记住，严格按照大五人格的定义生成符合角色大五人格的回复文本，不要出现对于性格的具体描述。"
    return prompts

prompts = prompt(agent_attributes_str,example)

csv_columns = list(agent_attributes.keys()) + ["text"]

with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_columns)

    for i, story_data in enumerate(data):
        story = story_data['story']
        teachable_agent = ConversableAgent(
            name="teachable_agent",
            system_message=f"1. 阅读{story}模拟给定信息的角色，根据基本信息{agent_attributes_str}，回答{question}",# The name is flexible, but should not contain spaces to work in group chat.
            llm_config=qwen2_7b  # Disable caching.
        )
        text = user.initiate_chat(teachable_agent, message=f"关于大五人格描述的定义为{description}，\n根据大五人格定义的解释，只回答符合{agent_attributes_str}中大五人格的回答，注意！只生成符合大五人格描述的回答，不要出现关于自身大五人格的水平描述！不需要对发言符合大五人格的特性进行总结。",clear_history=True)
        text_summary_list = text.summary.strip().split('\n\n')

        for i in range(len(text_summary_list)):  # 按顺序填入文本内容
            row_data = [agent_attributes[key] for key in csv_columns[:-1]]
            row_data.append(text_summary_list[i])
            writer.writerow(row_data)

