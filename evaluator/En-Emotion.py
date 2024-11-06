#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from scipy.special import softmax



tokenizer = AutoTokenizer.from_pretrained("/EN-emotion")
model = AutoModelForSequenceClassification.from_pretrained("/EN-emotion")
config = AutoConfig.from_pretrained("/EN-emotion")


csv_files = []

text_column = 'text'

for csv_file in csv_files:
    df = pd.read_csv(csv_file, encoding='gbk')

    # 用于存储每个文本的最高情感标签
    emotion_labels = []

    # 对每一行文本进行情感分析
    for index, row in df.iterrows():
        text = row[text_column]
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        # 获取最高情感标签
        highest_emotion_label = config.id2label[ranking[0]]
        emotion_labels.append(highest_emotion_label)

        print(f"Text: {text}")
        for i in range(scores.shape[0]):
            l = config.id2label[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i + 1}) {l} {np.round(float(s), 4)}")
        print("-" * 40)

    # 将最高情感标签添加到 DataFrame 中
    df['emotion'] = emotion_labels


    output_file = csv_file.replace('.csv', '_with_emotion.csv')
    df.to_csv(output_file, index=False, encoding='gbk')
