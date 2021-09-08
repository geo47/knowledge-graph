import re
import json


# credit: https://github.com/wang-h/bert-relation-classification/blob/master/utils.py
def clean_str(text):
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=<>]", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    #     text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"e - mail", "email", text)
    return text


# Replace first person and third person noun with user name from review
def replace_subject_entity(text, entity):
    text = re.sub(r"\bI\b|\bi\b|\bWe\b|\bwe\b", entity, text)
    return text


# Replace subject from the review to actual user name
def clean_reviews(restaurants_data, save_clean_data=False):
    rest_obj = dict()
    restaurants = restaurants_data
    for restaurant in restaurants:
        for review in restaurant['reviews']:
            review["text"] = clean_str(review["text"])
            review["text"] = replace_subject_entity(review["text"], review["name"])

    if save_clean_data:
        rest_obj['restaurants'] = restaurants
        with open('data/input/cleaned_reviews.json', 'w') as outfile:
            json.dump(rest_obj, outfile)