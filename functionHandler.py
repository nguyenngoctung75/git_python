import re
import json
import random

with open('categories.json', 'r', encoding='utf-8') as f:
    dataCate = json.load(f)

categories = [item["category"] for item in dataCate]

def extract_height_weight(text):
    # regex
    pattern = r"(\d+\.?\d*)\s?(m|cm|kg)"
    matches = re.findall(pattern, text)
    height, weight = None, None
    for value, unit in matches:
        value = float(value)
        if unit == "m" or unit == "cm":
            height = value * 100 if unit == "m" else value
        elif unit == "kg":
            weight = value
    return height, weight

def determine_size(text):
    height, weight = extract_height_weight(text)
    if not height or not weight:
        return "It seems like I don't have enough information about your weight or height. Could you please provide those details so I can offer more suitable suggestions?"
    if height <= 155 and weight <= 45:
        return "You should try size S"
    elif height <= 165 and weight <= 55:
        return "You should try size M"
    elif height <= 175 and weight <= 70:
        return "You should try size L"
    elif height <= 185 and weight <= 85:
        return "You should try size XL"
    else:
        return "You should try size XXL"

def extract_category(user_input, nlp):
    doc = nlp(user_input.lower())
    for token in doc:
        if token.text in categories:
            return token.text
    return None

def find_category(response, category):
    if category:
        response = response.replace("{category}", category)
        links = []
        for item in dataCate:
            if item["category"] == category:
                links = item["links"]
        
        if len(links) >= 3:
            links = random.sample(links, 3)
        
        links = "\n".join(links)
        return f"{response}\n{links}"
    else:
        return "Sorry, I couldn't find that product category. Please try again or try other categories."