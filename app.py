from flask import Flask, render_template, request, jsonify
import json
import random
import spacy
import math

app = Flask(__name__)

nlp = spacy.load("en_core_web_md")

# Load dữ liệu
with open('inputTest.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

patterns = []
responses = []
intents = []
D = 0
for item in data:
    for pattern in item['patterns']:
        patterns.append(pattern)
        intents.append(item['intent'])
        responses.append(item['responses'])
        D += 1

# Xử lý dữ liệu
def lemmatize_patterns(patterns):
    lemmatized = []
    for sentence in patterns:
        doc = nlp(sentence)
        lemmatized.append([token.lemma_ for token in doc if not token.is_stop])
    return lemmatized

patterns = lemmatize_patterns(patterns)
vocabulary = list(set(word for pattern in patterns for word in pattern))

idf_arr = [math.log(D / sum(1 for pattern in patterns if word in pattern)) for word in vocabulary]

def cal_tf_idf(document):
    tf_arr = [document.count(word) / len(document) for word in vocabulary]
    return [tf * idf for tf, idf in zip(tf_arr, idf_arr)]

tf_idf_all = [cal_tf_idf(pattern) for pattern in patterns]

def cal_p_Ck(intent):
    return intents.count(intent) / len(intents)

def cal_p_d_k(word, intent):
    sum_word = sum(tf_idf_all[idx][vocabulary.index(word)] for idx, item in enumerate(intents) if item == intent and word in vocabulary)
    sum_intent = sum(sum(tf_idf_all[idx]) for idx, item in enumerate(intents) if item == intent)
    return (sum_word + 1) / (sum_intent + len(vocabulary))

def get_intent(user_input):
    doc = nlp(user_input)
    user_words = [token.lemma_ for token in doc if not token.is_stop and token.lemma_ in vocabulary]
    probabilities = [(intent, math.log(cal_p_Ck(intent)) + sum(math.log(cal_p_d_k(word, intent)) for word in user_words)) for intent in set(intents)]
    return max(probabilities, key=lambda x: x[1])[0]

def get_response(user_input):
    res_intent = get_intent(user_input)
    idx = intents.index(res_intent)
    return random.choice(responses[idx])

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
