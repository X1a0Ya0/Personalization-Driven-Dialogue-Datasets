# -*- coding:utf-8 -*-

from autogen import ConversableAgent
import numpy as np
import pandas as pd
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
csv_files = []

# 初始化全局混淆矩阵
global_confusion_matrices = {
    "E_s": np.zeros((2, 2)),
    "A_s": np.zeros((2, 2)),
    "C_s": np.zeros((2, 2)),
    "N_s": np.zeros((2, 2)),
    "O_s": np.zeros((2, 2)),
}

# 处理每个CSV文件
for csv_file in csv_files:
    df = pd.read_csv(csv_file, encoding='gbk')

    # 提取相关列
    labels_df = df[['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']]

    # 获取对话列并处理
    conversation_column = df['text'].dropna()
    conversation_list = [' '.join(conversation_column[i:i + 5]) for i in range(0, len(conversation_column), 5)]

    questions = [
        "1. 你喜欢这个故事吗？",
        "2. 请概括一下故事的主要内容。",
        "3. 从故事中你学到了什么？",
        "4. 你赞同这个道理吗？说说你的看法。",
        "5. 读了这段故事，是否联想到了你自身的经历？"
    ]
    pattern = r'\d+\.\s*(.*?)(?=\d+\.|$)'
    bins = np.arange(0, 5.1, 0.1)
    # 创建字典来记录每个维度在每个区间的数量
    distribution = {
        "E_s": np.zeros(len(bins) - 1),
        "A_s": np.zeros(len(bins) - 1),
        "C_s": np.zeros(len(bins) - 1),
        "N_s": np.zeros(len(bins) - 1),
        "O_s": np.zeros(len(bins) - 1),
    }
    final_conversations = []

    for conversation in conversation_list:
        split_conversation = re.findall(pattern, conversation)
        combined_conversation = []
        for i in range(len(questions)):
            if i < len(split_conversation):
                combined_conversation.append(f" {split_conversation[i].strip()}")
        final_conversations.append(combined_conversation)

    # 初始化标签计数器
    label_counts = {
        "E_s": {"high": 0, "low": 0},
        "A_s": {"high": 0, "low": 0},
        "C_s": {"high": 0, "low": 0},
        "N_s": {"high": 0, "low": 0},
        "O_s": {"high": 0, "low": 0},
    }

    # 初始化正确预测计数器
    correct_counts = {
        "E_s": 0,
        "A_s": 0,
        "C_s": 0,
        "N_s": 0,
        "O_s": 0,
    }

    # 初始化Agent
    chat_glmagent = ConversableAgent(name="chat_glm", llm_config=qwen2_7b, system_message="你的任务是按照要求做事",
                                     human_input_mode='NEVER')

    question = "请用 1 到 5 分表示您是否同意以下每项陈述（1 = 非常不同意，2 = 不同意，3 = 既不同意也不不不同意，4 = 同意，5 = 非常同意）。我认为自己是一个 1）健谈 2）喜欢找别人的麻烦 3）工作认真 4）情绪低落，忧郁 5）有独创性，能提出新想法 6）拘谨 7）乐于助人，无私奉献 8）可能有点粗心大意 9）放松，能很好地应对压力 10）对许多不同的事物都充满好奇心 11）精力充沛 12）经常与人争吵 13）工作可靠 14）可能比较紧张 15）具有独创性，善于深思熟虑 16）能产生很大的热情 17）具有宽容的天性 18）做事有条理 19）有条理。倾向于杂乱无章 19) 常常忧虑 20) 想象力丰富 21) 倾向于安静 22) 普遍信任他人 23) 倾向于懒惰 倾向于懒惰 24) 情绪稳定，不容易生气 25) 富有创造力 26) 有自信的个性 27) 可能冷漠 28) 坚持不懈，直到完成任务 29) 可能喜怒无常 30) 重视艺术和审美体验 31) 有时害羞、拘谨 32) 几乎对所有人都体贴和友善 33) 做事效率高 34) 在紧张的情况下保持冷静 35) 喜欢例行工作 36) 性格外向、 37） 有时对他人无礼 38） 制定计划并付诸实施 39） 容易紧张 40） 喜欢思考，玩弄各种想法 41） 很少有艺术兴趣 42） 喜欢与他人合作 43） 容易分心 44） 在艺术、音乐或文学方面很成熟。"
    description = "其中大五人格特性的定义如下：Openness{[high]: 喜欢这个故事;[low]:不喜欢这个故事}\nConscientiousness{[high]: 对故事概括十分全面;[low]: 忽略故事的情节}\nExtraversion：{[high]:回答态度热情;[low]回答态度冷漠: }\nAgreeableness：{[high]:在交流过程多用'请'，'您'等文明用语;[low]: 不喜欢回答问题}\nNeuroticism：{[high]: 回答时出现用省略号表示的停顿;[low]:回答通顺}"
    description2 = "各个属性的解释：Openness{[high]: 喜欢这个故事;[low]:不喜欢这个故事}\nConscientiousness{[high]: 对故事概括十分全面;[low]: 忽略故事的情节}\nExtraversion：{[high]:回答态度热情;[low]回答态度冷漠: }\nAgreeableness：{[high]:在交流过程多用'请'，'您'等文明用语;[low]: 不喜欢回答问题}\nNeuroticism：{[high]: 回答时出现用省略号表示的停顿;[low]:回答通顺}"


    def generate_prompt(question, conversation, description2):
        prompt = f"根据对话{conversation}以及{description2}，对对话者进行评分，评分量表如下：{question}，注意!严格按照评分量表,只对量表问题进行评分，给出全部量表问题的结果，不需要对结果进行解释说明。评分结果例如:1. 1\n2. 3\n3. 2\n4. 2\n5. 2\n6. 2\n7. 2\n8. 2\n9. 1\n10. 2\n11. 3\n12. 2\n13. 5\n14. 1\n15. 2\n16. 2\n17. 1\n18. 5\n19. 2\n20. 1\n21. 2\n22. 4\n23. 4\n24. 2\n25. 1\n26. 2\n27. 5\n28. 2\n29. 1\n30. 1\n31. 1\n32. 2\n33. 3\n34. 2\n35. 2\n36. 1\n37. 5\n38. 2\n39. 1\n40. 1\n41. 4\n42. 2\n43. 4\n44. 2"
        return prompt

    confusion_matrices = {}
    score_results = []

    for idx, conversation in enumerate(final_conversations):
        prompt = generate_prompt(question, conversation, description)
        prompt = [{"role": "user", "content": f"{prompt}"}]
        response = chat_glmagent.generate_reply(messages=prompt, sender=chat_glmagent)
        if not isinstance(response, str):
            response = str(response)
        scores = re.findall(r'\d+', response)
        scores = [int(score) for score in scores]
        scores = scores[1::2]

        if len(scores) == 44 and all(score <= 5 for score in scores):
            E_s = (scores[0] + (5 - scores[5]) + scores[10] + (5 - scores[20]) + scores[25] + (5 - scores[30]) + scores[
                36]) / 7
            A_s = (scores[6] + (5 - scores[1]) + (5 - scores[11]) + scores[16] + scores[21] + (5 - scores[26]) + scores[
                31] + (5 - scores[36]) + scores[41]) / 9
            C_s = (scores[2] + (5 - scores[7]) + scores[12] + (5 - scores[17]) + (5 - scores[22]) + scores[27] + scores[
                32] + scores[37] + (5 - scores[42])) / 9
            N_s = (scores[3] + (5 - scores[8]) + scores[13] + scores[18] + (5 - scores[23]) + scores[28] + scores[38] + (
                        5 - scores[33])) / 8
            O_s = (scores[4] + scores[9] + scores[14] + scores[19] + scores[24] + scores[29] + (5 - scores[34]) + scores[
                39] + (5 - scores[40]) + scores[43]) / 10

            score_labels = {
                "E_s": "high" if E_s >= 3 else "low",
                "A_s": "high" if A_s > 3.3  else "low",
                "C_s": "high" if C_s >=3.0 else "low",
                "N_s": "high" if N_s > 2.1 else "low",
                "O_s": "high" if O_s >= 3. else "low",
            }
            distribution["E_s"] += np.histogram(E_s, bins=bins)[0]
            distribution["A_s"] += np.histogram(A_s, bins=bins)[0]
            distribution["C_s"] += np.histogram(C_s, bins=bins)[0]
            distribution["N_s"] += np.histogram(N_s, bins=bins)[0]
            distribution["O_s"] += np.histogram(O_s, bins=bins)[0]
            score_results.append(score_labels)

            # 比较预测结果与实际标签
            actual_labels = {
                "E_s": labels_df.loc[idx, 'Extraversion'],
                "A_s": labels_df.loc[idx, 'Agreeableness'],
                "C_s": labels_df.loc[idx, 'Conscientiousness'],
                "N_s": labels_df.loc[idx, 'Neuroticism'],
                "O_s": labels_df.loc[idx, 'Openness'],
            }

            for label, value in score_labels.items():
                label_counts[label][value] += 1
                if value == actual_labels[label]:
                    correct_counts[label] += 1

    for label, counts in label_counts.items():
        print(f"{label}: high={counts['high']}, low={counts['low']}")

    for label in label_counts:
        # True Positive: 实际为 high 且预测为 high 的数量
        if actual_labels[label] == 'high':
            tp = label_counts[label]['high']
        else:
            tp = 0

        # False Negative: 实际为 high 但预测为 low 的数量
        if actual_labels[label] == 'high':
            fn = label_counts[label]['low']
        else:
            fn = 0

        # False Positive: 实际为 low 但预测为 high 的数量
        if actual_labels[label] == 'low':
            fp = label_counts[label]['high']
        else:
            fp = 0

        # True Negative: 实际为 low 且预测为 low 的数量
        if actual_labels[label] == 'low':
            tn = label_counts[label]['low']
        else:
            tn = 0

        confusion_matrices[label] = np.array([[tp, fn], [fp, tn]])

    # 累加到全局混淆矩阵
    for label in global_confusion_matrices:
        global_confusion_matrices[label] += confusion_matrices[label]
    for label, matrix in global_confusion_matrices.items():
        print(f"Final confusion matrix for {label}:\n{matrix}")
    final_metrics = {}

    for label, matrix in global_confusion_matrices.items():
        tp, fn = matrix[0]  # 读取 True Positive 和 False Negative
        fp, tn = matrix[1]  # 读取 False Positive 和 True Negative

        # 计算 Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

        # 计算 Recall
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        # 计算 Precision
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0

        # 计算 F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        # 保存结果
        final_metrics[label] = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1_score": f1
        }

# 打印最终的混淆矩阵

# 计算最终的指标

# 打印最终的指标
for label, metric in final_metrics.items():
    print(f"Final metrics for {label}:")
    print(f"  Accuracy: {metric['accuracy']:.2f}")
    print(f"  Recall: {metric['recall']:.2f}")
    print(f"  Precision: {metric['precision']:.2f}")
    print(f"  F1 Score: {metric['f1_score']:.2f}")

