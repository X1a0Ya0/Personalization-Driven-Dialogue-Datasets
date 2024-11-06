##Chinese
question = "请用 1 到 5 分表示您是否同意以下每项陈述（1 = 非常不同意，2 = 不同意，3 = 既不同意也不不不同意，4 = 同意，5 = 非常同意）。我认为自己是一个 1）健谈 2）喜欢找别人的麻烦 3）工作认真 4）情绪低落，忧郁 5）有独创性，能提出新想法 6）拘谨 7）乐于助人，无私奉献 8）可能有点粗心大意 9）放松，能很好地应对压力 10）对许多不同的事物都充满好奇心 11）精力充沛 12）经常与人争吵 13）工作可靠 14）可能比较紧张 15）具有独创性，善于深思熟虑 16）能产生很大的热情 17）具有宽容的天性 18）做事有条理 19）有条理。倾向于杂乱无章 19) 常常忧虑 20) 想象力丰富 21) 倾向于安静 22) 普遍信任他人 23) 倾向于懒惰 倾向于懒惰 24) 情绪稳定，不容易生气 25) 富有创造力 26) 有自信的个性 27) 可能冷漠 28) 坚持不懈，直到完成任务 29) 可能喜怒无常 30) 重视艺术和审美体验 31) 有时害羞、拘谨 32) 几乎对所有人都体贴和友善 33) 做事效率高 34) 在紧张的情况下保持冷静 35) 喜欢例行工作 36) 性格外向、 37） 有时对他人无礼 38） 制定计划并付诸实施 39） 容易紧张 40） 喜欢思考，玩弄各种想法 41） 很少有艺术兴趣 42） 喜欢与他人合作 43） 容易分心 44） 在艺术、音乐或文学方面很成熟。"
description = "各个属性的解释：Openness{[high]: 喜欢这个故事;[low]:不喜欢这个故事}\nConscientiousness{[high]: 对故事概括十分全面;[low]: 忽略故事的情节}\nExtraversion：{[high]:回答态度热情;[low]回答态度冷漠: }\nAgreeableness：{[high]:在交流过程多用'请'，'您'等文明用语;[low]: 不喜欢回答问题}\nNeuroticism：{[high]: 回答时出现用省略号表示的停顿;[low]:回答通顺}"

def generate_prompt(question, conversation, description):
    prompt = f"根据对话{conversation}以及{description}，对对话者进行评分，评分量表如下：{question}，注意!严格按照评分量表,只对量表问题进行评分，给出全部量表问题的结果，不需要对结果进行解释说明。评分结果例如:1. 1\n2. 3\n3. 2\n4. 2\n5. 2\n6. 2\n7. 2\n8. 2\n9. 1\n10. 2\n11. 3\n12. 2\n13. 5\n14. 1\n15. 2\n16. 2\n17. 1\n18. 5\n19. 2\n20. 1\n21. 2\n22. 4\n23. 4\n24. 2\n25. 1\n26. 2\n27. 5\n28. 2\n29. 1\n30. 1\n31. 1\n32. 2\n33. 3\n34. 2\n35. 2\n36. 1\n37. 5\n38. 2\n39. 1\n40. 1\n41. 4\n42. 2\n43. 4\n44. 2"
    return prompt

def generate_prompt(conversation):
    prompt = f"根据{conversation}的内容，请分别判断回答的流利性和连贯性，你只需要回答是否可读和是否相关。每段对话按同一种回答格式。流利性：衡量回答是否可读，回答应该是人类可以理解的句子;连贯性：评估回答和问题的相关性，回答应该与问题的主题一致，而不是与主题无关;"
    return prompt

def prompt(trait,example):
    prompts= f"根据{trait}的属性，请直接给出问题的答案。符合角色大五人格的文本可以参考一下例子，例如：{example}。记住，严格按照大五人格的定义生成符合角色大五人格的回复文本，不要出现对于性格的具体描述。"
    return prompts

## English
question = "Please indicate your agreement with each of the following statements on a scale from 1 to 5 (1 = ‘Strongly disagree’, 2 = ‘Disagree’, 3 = ‘Neither agree or disagree’, 4 = ‘Agree’, and 5 = ‘Strongly agree’). I see myself as someone who...: 1) Is talkative 2) Tends to find fault with others 3) Does a thorough job 4) Is depressed, blue 5) Is original, comes up with new ideas 6) Is reserved 7) Is helpful and unselfish with others 8) Can be somewhat careless 9) Is relaxed, handles stress well 10) Is curious about many different things 11) Is full of energy 12) Starts quarrels with others 13) Is a reliable worker 14) Can be tense 15) Is ingenious, a deep thinker 16) Generates a lot of enthusiasm 17) Has a forgiving nature 18) Tends to be disorganized 19) Worries a lot 20) Has an active imagination 21) Tends to be quiet 22) Is generally trusting 23) Tends to be lazy 24) Is emotionally stable, not easily upset 25) Is inventive 26) Has an assertive personality 27) Can be cold and aloof 28) Perseveres until the task is finished 29) Can be moody 30) Values artistic, aesthetic experiences 31) Is sometimes shy, inhibited 32) Is considerate and kind to almost everyone 33) Does things efficiently 34) Remains calm in tense situations 35) Prefers work that is routine 36) Is outgoing, sociable 37) Is sometimes rude to others 38) Makes plans and follows through with them 39) Gets nervous easily 40) Likes to reflect, play with ideas 41) Has few artistic interests 42) Likes to cooperate with others 43) Is easily distracted 44) Is sophisticated in art, music, or literature"
description2 = "Explanation of each attribute: Openness {[high]: like this story; [low]: don't like this story}\nConscientiousness {[high]: give a comprehensive summary of the story; [low]: ignore the plot of the story}\nExtraversion: {[high]: answer with enthusiasm; [low] answer with indifference}\nAgreeableness: {[high]: use polite words  during communication; [low]: don't like to answer questions}\nNeuroticism: {[high]: pauses indicated by '...' when answering; [low]: answer fluently}"

def generate_prompt(question, conversation, description2):
    prompt = f"Based on the {conversation} and {description2}, rate the interlocutors's personality traits. The scoring scale is as follows: {question}. Note! Strictly follow the scoring scale and only score the scale questions. The results of all scale questions are given. Try to think about the basis of the rating without giving an explanation. Give answers in the order of the questionnaire, for example: 1. 1\n2. 3\n3. 2\n4. 2\n5. 2\n6. 2\n7. 2\n8. 2\n9. 1\n10. 2\n11. 3\n12. 2\n13. 5\n14. 1\n15. 2\n16. 2\n17. 1\n18. 5\n19. 2\n20. 1\n21. 2\n22. 4\n23. 4\n24. 2\n25. 1\n26. 2\n27. 5\n28. 2\n29. 1\n30. 1\n31. 2\n32. 2\n33. 3\n34. 2\n35. 2\n36. 1\n37. 5\n38. 2\n39. 1\n40. 1\n41. 4\n42. 2\n43. 4\n44. 2,"
    return prompt

def generate_prompt(conversation):
    prompt = f"Base on the {conversation}, Please evaluate the answers based on fluency and coherency. You only need to determine if they are readable andrelevant. Fluency: measures whether the answer is readable. The response should be in sentences that humans can understand. Coherency: assesses the relevance of the answer to the question. The response should be on-topic with the question’s theme rather than off-topic;"
    return prompt

def prompt(trait,example):
    prompts= f"Based on the {trait}, please give the answer to the question directly. For text that conforms to the speaker's Big Five personality traits, you can refer to the following example, such as: {example}. Remember, strictly follow the definition of the Big Five personality to generate a response text that conforms to the speaker's Big Five personality traits, and do not include a specific description of the personality."
    return prompts