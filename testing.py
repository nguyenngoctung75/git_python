import json
import random
import spacy
import math

nlp = spacy.load("en_core_web_md")

# Bước 1: Chuẩn bị dữ liệu
with open('inputTest.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

patterns = []
responses = []
intents = []
D = 0 # tổng số document
for item in data:
    for pattern in item['patterns']:
        patterns.append(pattern)
        intents.append(item['intent'])
        responses.append(item['responses'])
        D += 1


# Bước 2: Xử lý ngôn ngữ tự nhiên: loại bỏ stop word và đưa từ về dạng nguyên mẫu
# Trả về mảng 2 chiều gồm các "câu" đã qua xử lý
def lemmatize_patterns(patterns):
    lemmatized = []
    for sentence in patterns:
        doc = nlp(sentence)
        # .lemma_: đưa từ về dạng nguyên mẫu (Ex: did => do)
        # .is_stop: kiểm tra stop words 
        lemmatized.append([token.lemma_ for token in doc if not token.is_stop])
    return lemmatized

patterns = lemmatize_patterns(patterns)

# từ điển các từ có trong dữ liệu huấn luyện
vocabulary = []
for item1 in patterns:
    for item2 in item1:
        vocabulary.append(item2)
vocabulary = list(set(vocabulary))
# print(vocabulary)

# vectorization
def cal_idf():
    tmp = []
    for word in vocabulary:
        cnt = 0
        for item in patterns:
            if word in item:
                cnt += 1
        tmp.append(math.log(D/cnt))
    return tmp
idf_arr = cal_idf()

def cal_tf_idf(document):
    # cal_tf:
    tf_arr = []
    n = len(document)
    for word in vocabulary:
        cnt = document.count(word)
        tf_arr.append(cnt/n)
    
    # cal_tf_idf: tf-idf = tf*idf
    tf_idf = []
    for item1, item2 in zip(tf_arr, idf_arr):
        tf_idf.append(item1*item2)

    return tf_idf

# ma trận tf-idf của toàn bộ các document trong bộ dữ liệu huấn luyện
tf_idf_all = []
for document in patterns:
    tmp = cal_tf_idf(document)
    tf_idf_all.append(tmp)

# mỗi intent là một lớp, nhiệm vụ của mình là phân lớp dữ liệu mới dựa trên xác suất
# p_Ck: xác suất của 1 intent trong bộ dữ liệu huấn luyện
def cal_p_Ck(intent):
    return intents.count(intent) / len(intents)

# p_dk: xác suất có điều kiện p(xd|Ck)
def cal_p_d_k(word, intent):
    sum_word = 0
    sum_intent = 0

    for item in intents:
        if item == intent:
            if word in vocabulary:
                idx = intents.index(item)
                sum_word += tf_idf_all[idx][vocabulary.index(word)]
                sum_intent += sum(tf_idf_all[idx])
    
    return (sum_word + 1)/ (sum_intent + len(vocabulary))

# trả về intent dự đoán
def get_intent(user_input):
    doc = nlp(user_input)
    user_words = [token.lemma_ for token in doc if not token.is_stop and token.lemma_ in vocabulary]

    probabilities = []
    for intent in set(intents):
        p_ck = math.log(cal_p_Ck(intent))
        p_d_k = 0
        for word in user_words:
            p_d_k += math.log(cal_p_d_k(word, intent))
        
        probabilities.append((intent, p_ck + p_d_k))

    # trả về intent có xác suất tính được lớn nhất
    return max(probabilities, key=lambda x:x[1])[0]

# trả về câu trả lời ngẫu nhiên trong tập câu trả lời có intent tương ứng
def get_response(user_input):
    res_intent = get_intent(user_input)
    idx = intents.index(res_intent)
    return random.choice(responses[idx])

while True:
    user_input = input("Bạn: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = get_response(user_input)
    print("Chatbot:", response)
    



    

