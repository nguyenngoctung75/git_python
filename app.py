from flask import Flask, render_template, request, jsonify
import json
import random
import spacy
import math

import functionHandler as fh

app = Flask(__name__)

nlp = spacy.load("en_core_web_md")

# Load dữ liệu
with open('trainData.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

patterns = [item['patterns'] for item in data]
responses = [item['responses'] for item in data]
intents = [item['intent'] for item in data]

D = 0 # number of documents
for item1 in patterns:
    D += len(item1)

def lemmatize_sentence(sentence):
    doc = nlp(sentence)
    return [token.lemma_ for token in doc]

# lemmatize function
def lemmatize_patterns(patterns):
    lemmatized = []
    for pattern in patterns:
        tmp = []
        for sentence in pattern:
            tmp.append(lemmatize_sentence(sentence))

        lemmatized.append(tmp)
    return lemmatized

patterns = lemmatize_patterns(patterns)

# set vocabulary
vocabulary = {word for item1 in patterns for item2 in item1 for word in item2}
# print(vocabulary)

def vectorized(list_word):
    tmp = []
    size = len(list_word)
    for word in vocabulary:
        tf = list_word.count(word)/size
        cnt = sum(1 for item1 in patterns for item2 in item1 if word in item2)
        idf = math.log(D/(cnt+1))
        tmp.append(tf*idf)
    return tmp

patterns_vectorized = []
for item1 in patterns:
    tmp = []
    for item2 in item1:
        tmp.append(vectorized(item2))
    patterns_vectorized.append(tmp)

def p_Ck(i):
    return len(patterns[i]) / D

def p_dk(i, k):
    sum_xid = sum(vt[k] for vt in patterns_vectorized[i])
    all_xi = sum(sum(vt) for vt in patterns_vectorized[i])

    return (1e-06 + sum_xid)/(all_xi + 1e-06 * len(vocabulary))

def get_intent(user_input):
    user_input = lemmatize_sentence(user_input)
    input_vector = vectorized(user_input)
    # print(input_vector)

    if sum(input_vector) == 0: 
        return None

    probabilities = []
    i = 0
    for intent in intents:
        k = 0
        tmp = 0
        for xd in input_vector:
            pdk = p_dk(i,k)
            tmp += xd*math.log(pdk)
            k += 1
        pck = p_Ck(i)
        pck_x = 10 ** ( math.log(pck) + tmp )
        probabilities.append((intent, pck_x))
        i += 1

    predict_intent = max(probabilities, key=lambda x:x[1])[0]
    # print(predict_intent)
    return predict_intent

def get_response(user_input):
    predict_intent = get_intent(user_input)
    if predict_intent:
        if predict_intent == "size":
            return fh.determine_size(user_input)
        
        i = intents.index(predict_intent)
        response = random.choice(responses[i])
        if predict_intent == "product_search":
            category = fh.extract_category(user_input, nlp)
            return fh.find_category(response, category)
        
        return response
    else:
        return "I can't answer this question. Please wait until the shop owner comes back!"

# while True:
#     user_input = input("Bạn: ")
#     if user_input.lower() in ['exit', 'quit']:
#         break
#     response = get_response(user_input)
#     print("Chatbot:", response)

# API để nhận câu hỏi từ người dùng và trả về câu trả lời
@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': "Xin hãy nhập câu hỏi!"})
    response = get_response(user_input)
    return jsonify({'response': response})

# Giao diện web
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
