import glob
import argparse
import logging
import pickle
import re
import json
import csv
from pprint import pprint

import pandas as pd

import spacy
from spacy import displacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from openie import StanfordOpenIE
from extract_entity import use_nltk_ner, use_spacy_ner, use_stanford_ner, use_allen_ner
from extract_relation import AllanRE
from resolve_coreference import AllenCR, StanfordCR, SpacyCR
from utils.spell_check import levenshtein_ratio_and_distance


from IPython.core.display import display, HTML
from libs.gpr_pub import visualization


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
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def replace_subject_entity(text, entity):
    text = re.sub(r"\bI\b|\bi\b|\bWe\b|\bwe\b", entity, text)
    return text

# def get_compound_pairs(text_doc, verbose=False):
#
#     nlp = spacy.load('en_core_web_sm')
#     doc = nlp(text_doc)
#     """Return tuples of (multi-noun word, adjective or verb) for document."""
#     compounds = [tok for tok in doc if tok.dep_ == 'compound'] # Get list of compounds in doc
#     compounds = [c for c in compounds if c.i == 0 or doc[c.i - 1].dep_ != 'compound'] # Remove middle parts of compound nouns, but avoid index errors
#     tuple_list = []
#     if compounds:
#         for tok in compounds:
#             pair_item_1, pair_item_2 = (False, False) # initialize false variables
#             noun = doc[tok.i: tok.head.i + 1]
#             pair_item_1 = noun
#             # If noun is in the subject, we may be looking for adjective in predicate
#             # In simple cases, this would mean that the noun shares a head with the adjective
#             if noun.root.dep_ == 'nsubj':
#                 adj_list = [r for r in noun.root.head.rights if r.pos_ == 'ADJ']
#                 if adj_list:
#                     pair_item_2 = adj_list[0]
#                 if verbose == True: # For trying different dependency tree parsing rules
#                     print("Noun: ", noun)
#                     print("Noun root: ", noun.root)
#                     print("Noun root head: ", noun.root.head)
#                     print("Noun root head rights: ", [r for r in noun.root.head.rights if r.pos_ == 'ADJ'])
#             if noun.root.dep_ == 'dobj':
#                 verb_ancestor_list = [a for a in noun.root.ancestors if a.pos_ == 'VERB']
#                 if verb_ancestor_list:
#                     pair_item_2 = verb_ancestor_list[0]
#                 if verbose == True: # For trying different dependency tree parsing rules
#                     print("Noun: ", noun)
#                     print("Noun root: ", noun.root)
#                     print("Noun root head: ", noun.root.head)
#                     print("Noun root head verb ancestors: ", [a for a in noun.root.ancestors if a.pos_ == 'VERB'])
#             if pair_item_1 and pair_item_2:
#                 tuple_list.append((pair_item_1, pair_item_2))
#     return tuple_list

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
ATTRS = ["acomp"]
POSS_NOUN = ["NOUN", "PROPN", "X"]


def get_compounds(tokens):
    compounds = []
    lefts = list(tokens.lefts)
    compounds.extend([tok for tok in lefts if tok.dep_ == 'compound'])
    return compounds


def get_subject_compound(v):
    subs = []
    compounds = []
    for tok in v.lefts:
        if tok.dep_ in SUBJECTS and tok.pos_ != "DET":
            compounds = get_compounds(tok)
            compounds.extend([tok])
    if compounds:
        subs.extend(compounds)
    return subs


def get_object_compound(v):
    objs = []
    compounds = []
    for tok in v.rights:
        if tok.dep_ in OBJECTS:
            compounds = get_compounds(tok)
            compounds.extend([tok])
    if compounds:
        objs.extend(compounds)
    return objs


def get_attribute_compound(av):
    objs = []
    compounds = []
    for tok in av.rights:
        if tok.dep_ in ATTRS:
            compounds = get_compounds(tok)
            compounds.extend([tok])
    if compounds:
        objs.extend(compounds)
    return objs


def get_prep_object_compound(p):
    p_objs = []
    compounds = []
    for tok in p.rights:
        if tok.dep_ == 'pobj':
            compounds = get_compounds(tok)
            compounds.extend([tok])
    if compounds:
        p_objs.extend(compounds)
    return p_objs


def get_prep(p):
    prep = False
    for tok in p.rights:
        if tok.dep_ == 'prep':
            prep = tok
    return prep


def get_lexical_triplets_pairs(text_doc, verbose=False):

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text_doc)

    print([(ent.text, ent.label_) for ent in doc.ents])

    triplets = []

    verbs = [tok for tok in doc if tok.pos_ == "VERB"]

    # getting sub, verb, obj triples
    # Angela, visit, silver spoon restaurant
    for v in verbs:
        subs = get_subject_compound(v)
        objs = get_object_compound(v)

        if subs and objs:
            triplets.append([' '.join(str(i) for i in subs), v, ' '.join(str(i) for i in objs)])

    # getting sub, verb_prep, p_obj triples
    # Angela, visit_with, Angela's friends
    for v in verbs:
        subs = get_subject_compound(v)
        prep = get_prep(v)

        p_objs = False
        if prep:
            p_objs = get_prep_object_compound(prep)

        if subs and p_objs:
            triplets.append([' '.join(str(i) for i in subs), str(v)+'_'+str(prep), ' '.join(str(i) for i in p_objs)])

    # getting sub, possession, obj triples
    # Silver spoon restaurant, has, Chicken biryani
    subs, objs = (False, False)
    poss_nouns = [tok for tok in doc if tok.pos_ in POSS_NOUN]
    for n in poss_nouns:
        children = list(n.children)
        for child in children:
            if child.dep_ == 'poss' and child.pos_ in POSS_NOUN:
                compounds = get_compounds(child)
                compounds.extend([child])
                subs = compounds

                compounds = get_compounds(n)
                compounds.extend([n])
                objs = compounds

                if subs and objs:
                    triplets.append([' '.join(str(i) for i in subs), 'has', ' '.join(str(i) for i in objs)])

    # getting sub, aux, complementary object triples
    # food, are, good
    subs, objs = (False, False)
    aux_verbs = [tok for tok in doc if tok.pos_ == "AUX"]
    for av in aux_verbs:
        subs = get_subject_compound(av)
        objs = get_attribute_compound(av)

        if subs and objs:
            triplets.append([' '.join(str(i) for i in subs), av, ' '.join(str(i) for i in objs)])

    pprint(triplets)
    return triplets


def split_sentence(text):
    '''
    splits review into a list of sentences using spacy's sentence parser
    '''
    nlp = spacy.load('en_core_web_sm')
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


def triple_pruning(triples, ner_dict):
    entity_set = set(ner_dict.keys())
    final_triples = []

    for row, col in triples.iterrows():
        col['subject'] = col['subject'].strip()

        # check if Named Entity in subject sentence fragment
        # found_entity = False
        # for named_entity in entity_set:
        #     if named_entity in col['subject']:
        #         col['subject'] = named_entity
        #         found_entity = True
        #
        # if found_entity:
        final_triples.append(('Node', col['subject'], col['relation'], 'Node', col['object']))

    triple_df = pd.DataFrame(final_triples, columns=['Type1', 'Entity1', 'Relationship', 'Type2', 'Entity2']).drop_duplicates()
    return triple_df


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--use_ner', action='store_true', help="Resolve Co-reference")
    # parser.add_argument('--ner_type', type=str, default="allen", help="Possible values (spacy|nltk|stanford|allen) "
    #                                                                   "Used only if --use_ner is true")
    # parser.add_argument('--use_cr', action='store_true', default=False, help="Co-reference Resolution")
    # parser.add_argument('--cr_type', type=str, default="allen", help="Possible values (spacy|stanford|allen) Used "
    #                                                                  "only if --use_cr is true")
    # parser.add_argument('--use_re', action='store_true', default=False, help="Relation Extraction")
    # parser.add_argument('--core_nlp_path', type=str,
    #                     default="./stanford-corenlp-4.2.2",
    #                     help="Path of Core NLP library, required when --use_cr is true and --cr_type is stanford")
    # parser.add_argument('--verbose', action='store_true', default=True, help="Log")
    #
    # args = parser.parse_args()

    logger = logging.getLogger('main')
    logger.disabled = False

    verbose = False #args.verbose

    output_path = "./data/output/"
    ner_pickles_op = output_path + "ner/"
    cr_pickles_op = output_path + "cr/"

    dataset = r"data/input/restaurant_data.json"

    ''' read restaurant knowledgebase data '''
    knowledge_base_entities = r"data/kb/entities/"
    kb_restaurant_file = knowledge_base_entities + "restaurant.csv"
    kb_menu_file = knowledge_base_entities + "menu.csv"
    kb_general_file = knowledge_base_entities + "general.csv"
    kb_restaurant_aspects_file = knowledge_base_entities + "restaurant_aspects.csv"
    kb_menu_aspects_file = knowledge_base_entities + "menu_attrs.csv"

    entity_headers = ['Name', 'Label']

    ''' make dataframes for kb '''
    kb_restaurant_df = pd.read_csv(kb_restaurant_file, header=0, names=entity_headers)
    kb_menu_df = pd.read_csv(kb_menu_file, header=0, names=entity_headers)
    # kb_general_df = pd.read_csv(kb_general_file, header=0, names=['Name', 'Label'])
    kb_restaurant_aspects_df = pd.read_csv(kb_restaurant_aspects_file, header=0, names=entity_headers)
    kb_menu_aspects_df = pd.read_csv(kb_menu_aspects_file, header=0, names=entity_headers)
    print(kb_restaurant_df)

    ''' open restaurant dataset '''
    f = open(dataset, )
    dataset = json.load(f)
    f.close()

    ''' cleaning restaurant reviews data '''
    restaurants = []
    restaurants.extend(restaurant for restaurant in dataset["restaurants"])

    for restaurant in restaurants:
        for review in restaurant["reviews"]:
            print(review["review"])
            review["review"] = clean_str(review["review"])
            review["review"] = replace_subject_entity(review["review"], review["name"])
            print(review["review"])

    ''' Make reviews string '''
    for restaurant in restaurants:
        reviews = []
        for review in restaurant["reviews"]:
            reviews.append(review["review"])
        restaurant["review_str"] = " ".join(review for review in reviews)

    ''' coreference resolution '''
    allen_cr = AllenCR(True)
    predictor, nlp = allen_cr.load_models()

    for restaurant in restaurants:

        doc = restaurant["review_str"]
        clusters = predictor.predict(doc)['clusters']
        nlp_doc = nlp(doc)
        coref_resolved = allen_cr.improved_replace_corefs(nlp_doc, clusters)
        restaurant["review_str"] = coref_resolved

        # for review in restaurant["reviews"]:
        #     dummy_sentence = review["name"] + " visit " + restaurant["name"] + "."
        #     doc = dummy_sentence + " " + review["review"]
        #
        #     clusters = predictor.predict(doc)['clusters']
        #     nlp_doc = nlp(doc)
        #     # allen_cr.print_comparison(allen_cr.original_replace_corefs(nlp_doc, clusters),
        #     #                            allen_cr.improved_replace_corefs(nlp_doc, clusters))
        #     # coref_resolved = allen_cr.original_replace_corefs(nlp_doc, clusters)
        #     coref_resolved = allen_cr.improved_replace_corefs(nlp_doc, clusters)
        #
        #     # doc = coref_resolved
        #
        #     ## split doc into sentences and remove first sentence
        #     # nlp_small.add_pipe(nlp_small.create_pipe('sentencizer'))
        #     nlp_doc = nlp(coref_resolved)
        #     sentences = [sent.string.strip() for sent in nlp_doc.sents]
        #     ## remove dumy_sentence
        #     sentences.pop(0)
        #     rev = " ".join([sent for sent in sentences])
        #     review["review"] = rev
        #     print(rev)

    ''' make general NER list '''
    for restaurant in restaurants:
        doc = nlp(restaurant["review_str"])

        ner_dict = {}
        for x in doc.ents:
            entity_span = x.text

            has_restaurant_entity = False
            i = 0
            for restaurant in kb_restaurant_df['Name']:
                ratio = levenshtein_ratio_and_distance(restaurant.lower(), entity_span.lower(), ratio_calc=True)
                if ratio > 0.85:
                    has_restaurant_entity = True
                if has_restaurant_entity:
                    break
                i += 1

            if has_restaurant_entity:
                continue

            has_menu_entity = False
            i = 0
            for menu in kb_menu_df['Name']:
                ratio = levenshtein_ratio_and_distance(menu.lower(), entity_span.lower(), ratio_calc=True)
                # print(menu.lower(), entity_span.lower(), str(ratio))
                if ratio > 0.85:
                    has_menu_entity = True
                if has_menu_entity:
                    break
                i += 1

            if has_menu_entity:
                continue

            ner_dict[x.text] = x.label_

        ner_dict

        with open(kb_general_file, 'a') as f_object:
            dictwriter_object = csv.writer(f_object)
            for key, value in ner_dict.items():
                dictwriter_object.writerow([key, value])
            f_object.close()

            # opening the csv file in 'w' mode
        general_file = open(kb_general_file, 'w')

        with general_file:
            writer = csv.DictWriter(general_file, fieldnames=entity_headers)

            writer.writeheader()
            for key, value in ner_dict.items():
                writer.writerow({entity_headers[0]: key,
                                 entity_headers[1]: value})
        # reviews = []
        # if restaurant["reviews"] and len(restaurant["reviews"]) > 0:
        #
        #     for review in restaurant["reviews"]:
        #         reviews.append(review["review"])
        #
        #     reviews_str = " ".join(review for review in reviews)
        #     doc = nlp(reviews_str)
        #
        #     ner_dict = {}
        #     for x in doc.ents:
        #         entity_span = x.text
        #
        #         has_restaurant_entity = False
        #         i = 0
        #         for restaurant in kb_restaurant_df['Name']:
        #             ratio = levenshtein_ratio_and_distance(restaurant.lower(), entity_span.lower(), ratio_calc=True)
        #             if ratio > 0.85:
        #                 has_restaurant_entity = True
        #             if has_restaurant_entity:
        #                 break
        #             i += 1
        #
        #         if has_restaurant_entity:
        #             continue
        #
        #         has_menu_entity = False
        #         i = 0
        #         for menu in kb_menu_df['Name']:
        #             ratio = levenshtein_ratio_and_distance(menu.lower(), entity_span.lower(), ratio_calc=True)
        #             # print(menu.lower(), entity_span.lower(), str(ratio))
        #             if ratio > 0.85:
        #                 has_menu_entity = True
        #             if has_menu_entity:
        #                 break
        #             i += 1
        #
        #         if has_menu_entity:
        #             continue
        #
        #         ner_dict[x.text] = x.label_
        #
        #     ner_dict
        #
        #     with open(kb_general_file, 'a') as f_object:
        #         dictwriter_object = csv.writer(f_object)
        #         for key, value in ner_dict.items():
        #             dictwriter_object.writerow([key, value])
        #         f_object.close()
        #
        #         # opening the csv file in 'w' mode
        #     general_file = open(kb_general_file, 'w')
        #
        #     with general_file:
        #         writer = csv.DictWriter(general_file, fieldnames=entity_headers)
        #
        #         writer.writeheader()
        #         for key, value in ner_dict.items():
        #             writer.writerow({entity_headers[0]: key,
        #                              entity_headers[1]: value})
    # print(doc)

    with StanfordOpenIE() as client:

        for restaurant in restaurants:
            if restaurant["reviews"] and len(restaurant["reviews"]) > 0:

                triples = []
                text = restaurant["review_str"]
                tuple_pairs = get_lexical_triplets_pairs(text)
                # pairs = list(set(tuple(sub) for sub in tuple_pairs))
                # pairs.append(['silver spoon restaurant','like','chicken biryani'])
                # pairs.append(['services', 'are', 'good'])
                triples.extend(tuple_pairs)


                # for review in restaurant["reviews"]:
                #
                #     text = review["review"]
                #     # print('Text: %s.' % text)
                #
                #     # sentences = split_sentence(text)
                #     #
                #     # for sentence in sentences:
                #     #     # print("processing sentence: ")
                #     #     # print(sentence)
                #     #
                #     #     for triple in client.annotate(str(sentence)):
                #     #         tri = []
                #     #
                #     #         # print('|-', triple)
                #     #         tri.append(triple.get("subject"))
                #     #         tri.append(triple.get("relation"))
                #     #         tri.append(triple.get("object"))
                #     #         triples.append(tri)
                #     #
                #     # print(triples)
                #
                #     tuple_pairs = get_lexical_triplets_pairs(text)
                #     # pairs = list(set(tuple(sub) for sub in tuple_pairs))
                #     # pairs.append(['silver spoon restaurant','like','chicken biryani'])
                #     # pairs.append(['services', 'are', 'good'])
                #     triples.extend(tuple_pairs)

                df = pd.DataFrame(triples, columns=['subject', 'relation', 'object'])
                print(df)
                df.to_csv("data/output/kg/input_data.txt-out.csv", index=False)
                from create_kg import draw_kg

                draw_kg(df)

                graph = list()
                graph.append('digraph {')
                for er in tuple_pairs:
                    graph.append('"{}" -> "{}" [ label="{}" ];'.format(er[0], er[2], er[1]))
                graph.append('}')
                graph_image = 'graph.png'
                import os
                from subprocess import Popen
                from sys import stderr

                output_dir = os.path.join('.', os.path.dirname(graph_image))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                import tempfile

                out_dot = os.path.join(tempfile.gettempdir(), 'graph.dot')
                with open(out_dot, 'w') as output_file:
                    output_file.writelines(graph)

                command = 'dot -Tpng {} -o {}'.format(out_dot, graph_image)
                dot_process = Popen(command, stdout=stderr, shell=True)
                dot_process.wait()
                assert not dot_process.returncode, 'ERROR: Call to dot exited with a non-zero code status.'

                # graph_image = 'graph.png'
                # client.generate_graphviz_graph(text, graph_image)
                # print('Graph generated: %s.' % graph_image)
