# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from autogen import ConversableAgent
import json
import re
gemma2 =  {
    "config_list": [
    {
        "model":"gemma2",
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
import numpy as np
import pandas as pd
import re

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
        "1. Do you like this story?",
        "2. Please summarize the main content of the story.",
        "3. What did you learn from the story?",
        "4. Do you agree with this opinion,  Tell me what you think?",
        "5. When you read this story, do you think of your own experience?"
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
    chat_glmagent = ConversableAgent(name="chat_glm", llm_config=gpt, system_message="Your task is to evaluate the conversation", human_input_mode='NEVER')

    question = "Please indicate your agreement with each of the following statements on a scale from 1 to 5 (1 = ‘Strongly disagree’, 2 = ‘Disagree’, 3 = ‘Neither agree or disagree’, 4 = ‘Agree’, and 5 = ‘Strongly agree’). I see myself as someone who...: 1) Is talkative 2) Tends to find fault with others 3) Does a thorough job 4) Is depressed, blue 5) Is original, comes up with new ideas 6) Is reserved 7) Is helpful and unselfish with others 8) Can be somewhat careless 9) Is relaxed, handles stress well 10) Is curious about many different things 11) Is full of energy 12) Starts quarrels with others 13) Is a reliable worker 14) Can be tense 15) Is ingenious, a deep thinker 16) Generates a lot of enthusiasm 17) Has a forgiving nature 18) Tends to be disorganized 19) Worries a lot 20) Has an active imagination 21) Tends to be quiet 22) Is generally trusting 23) Tends to be lazy 24) Is emotionally stable, not easily upset 25) Is inventive 26) Has an assertive personality 27) Can be cold and aloof 28) Perseveres until the task is finished 29) Can be moody 30) Values artistic, aesthetic experiences 31) Is sometimes shy, inhibited 32) Is considerate and kind to almost everyone 33) Does things efficiently 34) Remains calm in tense situations 35) Prefers work that is routine 36) Is outgoing, sociable 37) Is sometimes rude to others 38) Makes plans and follows through with them 39) Gets nervous easily 40) Likes to reflect, play with ideas 41) Has few artistic interests 42) Likes to cooperate with others 43) Is easily distracted 44) Is sophisticated in art, music, or literature"
    description = "The definitions of the Big Five personality traits are as follows: Openness {[high]: like this story; [low]: don't like this story}\nConscientiousness {[high]: summarize the story very comprehensively; [low]: ignore the plot of the story}\nExtraversion: {[high]: answer with enthusiasm; [low] answer with indifference}\nAgreeableness: {[high]: use civilized terms during communication; [low]: don't like to answer questions}\nNeuroticism: {[high]: pauses indicated by ellipsis when answering; [low]: answer fluently}"
    description2 = "Explanation of each attribute: Openness {[high]: like this story; [low]: don't like this story}\nConscientiousness {[high]: give a comprehensive summary of the story; [low]: ignore the plot of the story}\nExtraversion: {[high]: answer with enthusiasm; [low] answer with indifference}\nAgreeableness: {[high]: use polite words  during communication; [low]: don't like to answer questions}\nNeuroticism: {[high]: pauses indicated by '...' when answering; [low]: answer fluently}"

    def generate_prompt(question, conversation, description2):
        prompt = f"Based on the {conversation} and {description2}, rate the interlocutors's personality traits. The scoring scale is as follows: {question}. Note! Strictly follow the scoring scale and only score the scale questions. The results of all scale questions are given. Try to think about the basis of the rating without giving an explanation. Give answers in the order of the questionnaire, for example: 1. 1\n2. 3\n3. 2\n4. 2\n5. 2\n6. 2\n7. 2\n8. 2\n9. 1\n10. 2\n11. 3\n12. 2\n13. 5\n14. 1\n15. 2\n16. 2\n17. 1\n18. 5\n19. 2\n20. 1\n21. 2\n22. 4\n23. 4\n24. 2\n25. 1\n26. 2\n27. 5\n28. 2\n29. 1\n30. 1\n31. 2\n32. 2\n33. 3\n34. 2\n35. 2\n36. 1\n37. 5\n38. 2\n39. 1\n40. 1\n41. 4\n42. 2\n43. 4\n44. 2,"
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
                "E_s": "high" if E_s >= 2.7 else "low",
                "A_s": "high" if A_s >= 2.8 else "low",
                "C_s": "high" if C_s > 2.5 else "low",
                "N_s": "high" if N_s > 2.1 else "low",
                "O_s": "high" if O_s >= 3.7 else "low",
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
        tp, fn = matrix[0]
        fp, tn = matrix[1]
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
