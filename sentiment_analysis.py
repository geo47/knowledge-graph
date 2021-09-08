import xml.etree.ElementTree as ET

import os
import pickle
from collections import Counter, defaultdict
import re
from pprint import pprint
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from skmultilearn.problem_transform import LabelPowerset


import spacy
import neuralcoref

import gensim

nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)


class BuildModel:

    def __init__(self, doc_path, lexicon_file_path, embedding_path, pickle_path):
        print("Initializing Model class...")
        self.doc_path = doc_path
        self.lexicon_file_path = lexicon_file_path
        self.embedding_path = embedding_path
        self.pickle_path = pickle_path

    def preprocess_doc(self):

        tree = ET.parse(self.doc_path)
        root = tree.getroot()

        # Use this dataframe for multilabel classification
        # Must use scikitlearn's multilabel binarizer

        labeled_reviews = []
        for sentence in root.findall("sentence"):
            entry = {}
            aterms = []
            aspects = []
            if sentence.find("aspectTerms"):
                for aterm in sentence.find("aspectTerms").findall("aspectTerm"):
                    aterms.append(aterm.get("term"))
            if sentence.find("aspectCategories"):
                for aspect in sentence.find("aspectCategories").findall("aspectCategory"):
                    aspects.append(aspect.get("category"))
            entry["text"], entry["terms"], entry["aspects"] = sentence[0].text, aterms, aspects
            labeled_reviews.append(entry)
        annotated_reviews_df = pd.DataFrame(labeled_reviews)
        print("there are", len(labeled_reviews), "reviews in this training set")

        # Create a new column for text whose pronouns have been replaced
        annotated_reviews_df["text_pro"] = annotated_reviews_df.text.map(lambda x: self.replace_pronouns(x))

        # Save annotated reviews
        annotated_reviews_df.to_pickle(self.pickle_path+'annotated_reviews_df.pkl')
        print(annotated_reviews_df.head())

    def prepare_model(self, annotated_reviews_df):
        # Convert the multi-labels into arrays
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(annotated_reviews_df.aspects)
        X = annotated_reviews_df.text_pro

        # Split data into train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0)

        # save the the fitted binarizer labels
        # This is important: it contains the how the multi-label was binarized, so you need to
        # load this in the next folder in order to undo the transformation for the correct labels.
        filename = self.pickle_path+'mlb.pkl'
        pickle.dump(mlb, open(filename, 'wb'))

        # LabelPowerset allows for multi-label classification
        # Build a pipeline for multinomial naive bayes classification
        text_clf = Pipeline([('vect', CountVectorizer(stop_words="english", ngram_range=(1, 1))),
                             ('tfidf', TfidfTransformer(use_idf=False)),
                             ('clf', LabelPowerset(MultinomialNB(alpha=1e-1))), ])
        text_clf = text_clf.fit(X_train, y_train)
        predicted = text_clf.predict(X_test)

        # Calculate accuracy
        accuracy = np.mean(predicted == y_test)
        print('accuracy: ', accuracy)

        # Test if SVM performs better
        from sklearn.linear_model import SGDClassifier
        text_clf_svm = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf-svm', LabelPowerset(
                                     SGDClassifier(loss='hinge', penalty='l2',
                                                   alpha=1e-3, max_iter=6, random_state=42)))])
        _ = text_clf_svm.fit(X_train, y_train)
        predicted_svm = text_clf_svm.predict(X_test)

        # Calculate accuracy
        svm_accuracy = np.mean(predicted_svm == y_test)
        print('svm_accuracy: ', svm_accuracy)

        # Train naive bayes on full dataset and save model
        text_clf = Pipeline([('vect', CountVectorizer(stop_words="english", ngram_range=(1, 1))),
                             ('tfidf', TfidfTransformer(use_idf=False)),
                             ('clf', LabelPowerset(MultinomialNB(alpha=1e-1))), ])
        text_clf = text_clf.fit(X, y)

        # save the model to disk
        filename = self.pickle_path+'naive_model.pkl'
        pickle.dump(text_clf, open(filename, 'wb'))

        # print dataframe
        # mlb.inverse_transform(predicted)
        pred_df = pd.DataFrame(
            {'text_pro': X_test,
             'pred_category': mlb.inverse_transform(predicted)
             })
        pd.set_option('display.max_colwidth', -1)
        print(pred_df.head())

    def load_df(self):
        # Read annotated reviews df, which is the labeled dataset for training
        # This is located in the pickled files folder
        annotated_reviews_df = pd.read_pickle(self.pickle_path+'annotated_reviews_df.pkl')
        print(annotated_reviews_df.head(3))
        return annotated_reviews_df

    def load_embeddings_and_model(self):

        # Setup nltk corpora path and Google Word2Vec location
        if not os.path.isfile(self.pickle_path + 'word2vec_google.pkl'):
            google_vec_file = self.embedding_path+'GoogleNews-vectors-negative300.bin'
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(google_vec_file, binary=True)
            pickle.dump(word2vec, open(self.pickle_path + "word2vec_google.pkl", 'wb'))

        # If above script has been run, load saved word embedding
        self.word2vec = pickle.load(open(self.pickle_path + "word2vec_google.pkl", 'rb'))

        # load the Multi-label binarizer from previous notebook
        self.mlb = pickle.load(open(self.pickle_path + "mlb.pkl", 'rb'))

        # load the fitted naive bayes model from previous notebook
        self.naive_model = pickle.load(open(self.pickle_path + "naive_model.pkl", 'rb'))

        return self.word2vec, self.mlb, self.naive_model

    def load_opinion_lexicon(self):
        # Load opinion lexicon
        neg_file = open(self.lexicon_file_path+"negative-words.txt", encoding="ISO-8859-1")
        pos_file = open(self.lexicon_file_path+"positive-words.txt", encoding="ISO-8859-1")
        neg = [line.strip() for line in neg_file.readlines()]
        pos = [line.strip() for line in pos_file.readlines()]
        opinion_words = neg + pos
        return  opinion_words, pos, neg

    # Define function for replacing pronouns using neuralcoref
    def replace_pronouns(self, text):
        doc = nlp(text)
        return doc._.coref_resolved

    def check_similarity(self, aspects, word):
        '''
        checks for word2vec similarity values between category word and the term
        returns most similar word
        '''
        similarity = []
        for aspect in aspects:
            similarity.append(self.word2vec.n_similarity([aspect], [word]))
        # set threshold for max value
        if max(similarity) > 0.30:
            return aspects[np.argmax(similarity)]
        else:
            return None

    def assign_term_to_aspect(self, aspect_sent, terms_dict, sent_dict, pred):
        '''
        function: takes in a sentiment dictionary and appends the aspect dictionary
        inputs: sent_dict is a Counter in the form Counter(term:sentiment value)
                aspect_sent is total sentiment tally
                terms_dict is dict with individual aspect words associated with sentiment
        output: return two types of aspect dictionaries:
                updated terms_dict and aspect_sent
        '''
        aspects = ['ambience', 'food', 'price', 'service']

        # First, check word2vec
        # Note: the .split() is used for the term because word2vec can't pass compound nouns
        for term in sent_dict:
            try:
                # The conditions for when to use the NB classifier as default vs word2vec
                if self.check_similarity(aspects, term.split()[-1]):
                    terms_dict[self.check_similarity(aspects, term.split()[-1])][term] += sent_dict[term]
                    if sent_dict[term] > 0:
                        aspect_sent[self.check_similarity(aspects, term.split()[-1])]["pos"] += sent_dict[term]
                    else:
                        aspect_sent[self.check_similarity(aspects, term.split()[-1])]["neg"] += abs(sent_dict[term])
                elif (pred[0] == "anecdotes/miscellaneous"):
                    continue
                elif (len(pred) == 1):
                    terms_dict[pred[0]][term] += sent_dict[term]
                    if sent_dict[term] > 0:
                        aspect_sent[pred[0]]["pos"] += sent_dict[term]
                    else:
                        aspect_sent[pred[0]]["neg"] += abs(sent_dict[term])
                # if unable to classify via NB or word2vec, then put them in misc. bucket
                else:
                    terms_dict["misc"][term] += sent_dict[term]
                    if sent_dict[term] > 0:
                        aspect_sent["misc"]["pos"] += sent_dict[term]
                    else:
                        aspect_sent["misc"]["neg"] += abs(sent_dict[term])
            except:
                print(term, "not in vocab")
                continue
        return aspect_sent, terms_dict

    def feature_sentiment(self, sentence):
        '''
        input: dictionary and sentence
        function: appends dictionary with new features if the feature did not exist previously,
                  then updates sentiment to each of the new or existing features
        output: updated dictionary
        '''

        sent_dict = Counter()
        sentence = nlp(sentence)
        opinion_words, pos, neg = self.load_opinion_lexicon()
        debug = 0
        for token in sentence:
            #    print(token.text,token.dep_, token.head, token.head.dep_)
            # check if the word is an opinion word, then assign sentiment
            if token.text in opinion_words:
                sentiment = 1 if token.text in pos else -1
                # if target is an adverb modifier (i.e. pretty, highly, etc.)
                # but happens to be an opinion word, ignore and pass
                if (token.dep_ == "advmod"):
                    continue
                elif (token.dep_ == "amod"):
                    sent_dict[token.head.text] += sentiment
                # for opinion words that are adjectives, adverbs, verbs...
                else:
                    for child in token.children:
                        # if there's a adj modifier (i.e. very, pretty, etc.) add more weight to sentiment
                        # This could be better updated for modifiers that either positively or negatively emphasize
                        if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in opinion_words):
                            sentiment *= 1.5
                        # check for negation words and flip the sign of sentiment
                        if child.dep_ == "neg":
                            sentiment *= -1
                    for child in token.children:
                        # if verb, check if there's a direct object
                        if (token.pos_ == "VERB") & (child.dep_ == "dobj"):
                            sent_dict[child.text] += sentiment
                            # check for conjugates (a AND b), then add both to dictionary
                            subchildren = []
                            conj = 0
                            for subchild in child.children:
                                if subchild.text == "and":
                                    conj = 1
                                if (conj == 1) and (subchild.text != "and"):
                                    subchildren.append(subchild.text)
                                    conj = 0
                            for subchild in subchildren:
                                sent_dict[subchild] += sentiment

                    # check for negation
                    for child in token.head.children:
                        noun = ""
                        if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in opinion_words):
                            sentiment *= 1.5
                        # check for negation words and flip the sign of sentiment
                        if (child.dep_ == "neg"):
                            sentiment *= -1

                    # check for nouns
                    for child in token.head.children:
                        noun = ""
                        if (child.pos_ == "NOUN") and (child.text not in sent_dict):
                            noun = child.text
                            # Check for compound nouns
                            for subchild in child.children:
                                if subchild.dep_ == "compound":
                                    noun = subchild.text + " " + noun
                            sent_dict[noun] += sentiment
                        debug += 1
        return sent_dict

    def classify_and_sent(self, sentence, aspect_sent, terms_dict):
        '''
        function: classify the sentence into a category, and assign sentiment
        note: aspect_dict is a parent dictionary with all the aspects
        input: sentence & aspect dictionary, which is going to be updated
        output: updated aspect dictionary
        '''
        # classify sentence with NB classifier
        predicted = self.naive_model.predict([sentence])
        pred = self.mlb.inverse_transform(predicted)

        # get aspect names and their sentiment in a dictionary form
        sent_dict = self.feature_sentiment(sentence)

        # try to categorize the aspect names into the 4 aspects in aspect_dict
        aspect_sent, terms_dict = self.assign_term_to_aspect(aspect_sent, terms_dict, sent_dict, pred[0])
        return aspect_sent, terms_dict

    def split_sentence(self, text):
        '''
        splits review into a list of sentences using spacy's sentence parser
        '''
        review = nlp(text)
        bag_sentence = []
        start = 0
        for token in review:
            if token.sent_start:
                bag_sentence.append(review[start:(token.i - 1)])
                start = token.i
            if token.i == len(review) - 1:
                bag_sentence.append(review[start:(token.i + 1)])
        return bag_sentence

    # Remove special characters using regex
    def remove_special_char(self, sentence):
        return re.sub(r"[^a-zA-Z0-9.',:;?]+", ' ', sentence)

    def review_pipe(self, review, aspect_sent,
                    terms_dict={'ambience': Counter(), 'food': Counter(), 'price': Counter(), 'service': Counter(),
                                'misc': Counter()}):
        review = self.replace_pronouns(review)
        sentences = self.split_sentence(review)
        for sentence in sentences:
            sentence = self.remove_special_char(str(sentence))
            aspect_sent, terms_dict = self.classify_and_sent(sentence.lower(), aspect_sent, terms_dict)
        return aspect_sent, terms_dict


def main():
    print("main")

    file_path = 'data/yelp/Restaurants_Train.xml'
    lexicon_file_path = 'data/opinion_lexicon-en/'
    embedding_path = 'embeddings/'
    pickle_path = 'pickled_files/'

    model = BuildModel(file_path, lexicon_file_path, embedding_path, pickle_path)
    if not os.path.isfile(pickle_path + 'annotated_reviews_df.pkl') or \
        not os.path.isfile(pickle_path + 'word2vec_google.pkl') or \
        not os.path.isfile(pickle_path + 'mlb.pkl') or \
        not os.path.isfile(pickle_path + 'naive_model.pkl'):
        model.preprocess_doc()
        annotated_reviews_df = model.load_df()
        model.prepare_model(annotated_reviews_df)
        # print(annotated_reviews_df)

    word2vec, mlb, naive_model = model.load_embeddings_and_model()

    # Log classes in multilabel binarizer used for the model
    print(mlb.classes_)

    # word embedding from word2vec will be used to supplement the naive bayes categorization
    # of aspect terms.
    print(word2vec.n_similarity(['food'], ["sushi"]))

    # test code for feature sentiment
    sentence = "I came here with my friends on a Tuesday night. The sushi here is amazing. Our waiter was very helpful, but the music was terrible."
    senti_dict = model.feature_sentiment(sentence)
    print("Future sentiment")
    print(senti_dict)

    # uncomment to visualize dependency words via spaCy's displacy feature
    # displacy.render(spacy(sentence), style='dep',jupyter=True)

    #######################################################
    # Uncomment the following part to run test cases      #
    #######################################################
    # test case 1
    # terms_dict = {'ambience': Counter(), 'food': Counter(), 'price': Counter(), 'service': Counter(),
    #               'misc': Counter()}
    # aspect_sent = {'ambience': Counter(), 'food': Counter(), 'price': Counter(), 'service': Counter(),
    #                'misc': Counter()}
    # review = "Our waiter was not very helpful, and the music was terrible."
    # aspects, terms = model.review_pipe(review, aspect_sent, terms_dict)
    # print("test case 1")
    # pprint(aspects)
    # pprint(terms)
    #
    # # test case 2
    # review = "top notch"
    # aspects, terms = model.review_pipe(review, aspect_sent, terms_dict)
    # print("test case 2")
    # pprint(aspects)
    # pprint(terms)
    #
    # for key in terms:
    #     if terms[key]:
    #         print(str(key), ":", terms[key])

    reviews = []
    # reviews.append("Angela usually visit silver spoon restaurant with Angela's friends. silver spoon "
    #                "restaurant's food quality is amazing and Angela like silver spoon restaurant's "
    #                "chicken biryani a lot. silver spoon restaurant's chicken biryani so delicious. silver spoon "
    #                "restaurant's food as well as service are all good.")
    review_file = 'data/yelp/restaurant_corpus_indo.txt'
    review = ''
    # sample_data_limit = 1000
    sample_data_limit = 10

    no_of_reviews = 0
    num_lines = sum(1 for line in open(review_file, 'r'))
    with open(review_file, 'r') as f:
        for line in tqdm(f, total=num_lines):
            if line and line != '\n':
                # print(line)
                review += line
            else:
                if review:
                    reviews.append(review)
                    no_of_reviews += 1
                review = ''

            if no_of_reviews >= sample_data_limit:
                break

    print(reviews)
    print(str(len(reviews)))

    terms_dict = {'ambience': Counter(), 'food': Counter(), 'price': Counter(), 'service': Counter(),
                  'misc': Counter()}
    aspect_sent = {'ambience': Counter(), 'food': Counter(), 'price': Counter(), 'service': Counter(),
                   'misc': Counter()}
    for review in reviews:
        aspect_sent, terms_dict = model.review_pipe(review, aspect_sent, terms_dict)

    print("aspect_sent")
    pprint(aspect_sent)

    print("terms_dict")
    pprint(terms_dict)

    # pickle the aspect terms and sentiment separately.
    # Modify this code for the restaurant of interest.
    pickle.dump(aspect_sent, open(pickle_path+"aspects_sentiment.pkl", 'wb'))
    pickle.dump(terms_dict, open(pickle_path+"terms_dict.pkl", 'wb'))

main()