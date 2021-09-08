
import json
import csv

from pprint import pprint

import pandas as pd

# from py2neo import Graph

from openie import StanfordOpenIE
from extract_entity import use_nltk_ner, use_spacy_ner, use_stanford_ner, use_allen_ner
from extract_relation import AllanRE
from resolve_coreference import AllenCR, StanfordCR, SpacyCR
from utils.spell_check import levenshtein_ratio_and_distance

from IPython.core.display import display, HTML
from libs.gpr_pub import visualization

import spacy
from module.ner.menu_ner import MenuNER
from module.lexicon.lexical_analyzer import SpacyLexicalAnalyzer
from module.processor.relation_processor import TripletProcessor

from module.neo4j.graph_db import GraphDB


output_path = "./data/output/"
ner_pickles_op = output_path + "ner/"
cr_pickles_op = output_path + "cr/"

''' read restaurant knowledgebase data '''
knowledge_base_entities = r"data/kb/entities/"
kb_restaurant_file = knowledge_base_entities + "restaurant.csv"
kb_menu_file = knowledge_base_entities + "menu.csv"
kb_menu_file1 = knowledge_base_entities + "menu1.csv"
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
# print(kb_restaurant_aspects_df)


''' open coref_resolved_reviews restaurant dataset '''
restaurant_coref_resolved_reviews_file = r"data/input/coref_resolved_reviews.json"
file = open(restaurant_coref_resolved_reviews_file,)
restaurant_cleaned_reviews = json.load(file)
file.close()


triplet_processor = TripletProcessor()
triplet_processor.init_kb_dict(kb_restaurant_file, kb_restaurant_aspects_file,  kb_menu_file)
triplet_processor.init_lexical_analyzer()

graph = GraphDB("bolt://localhost:7687", "neo4j", "erclab")

menu_ner = MenuNER()
spacy_lexical_analyzer = SpacyLexicalAnalyzer()
spacy_lexical_analyzer.load_spacy_models("small")
triples_df = pd.DataFrame()
nlp = spacy.load('en_core_web_sm')

rest_index=0
rest_count = len(restaurant_cleaned_reviews['restaurants'])
for restaurant in restaurant_cleaned_reviews['restaurants']:

    rest_index += 1
    print("Processing Restaurant "+str(rest_index)+"/"+str(rest_count))

    reviews = restaurant['reviews']

    rev_index=0
    rev_count = len(reviews)
    for review in reviews:

        rev_index +=1
        print("\t Processing Review "+str(rev_index)+"/"+str(rev_count))

#         if rev_index < 6:
#             continue

        ner_dict = {}
        doc = nlp(review["text"])

        for x in doc.ents:
            entity_span = x.text

            has_restaurant_entity = False
            i = 0
            for kb_restaurant in kb_restaurant_df['Name']:
                ratio = levenshtein_ratio_and_distance(kb_restaurant.lower(), entity_span.lower(), ratio_calc=True)
                if ratio > 0.90:
                    has_restaurant_entity = True
                if has_restaurant_entity:
                    break
                i += 1

            if has_restaurant_entity:
                continue

            if x.label_ not in ["CARDINAL", "ORDINAL"]:
                ner_dict[x.text] = x.label_

        text = review["text"]
        tuple_pairs = spacy_lexical_analyzer.get_lexical_triplets_pairs(text)
        tuple_pairs_df = pd.DataFrame(tuple_pairs, columns=['subject', 'relation', 'object'])
        # pairs = list(set(tuple(sub) for sub in tuple_pairs))
        rest = {"rest_id": restaurant["rest_id"], "name": restaurant["name"]}
        tuple_pairs_prune = triplet_processor.process_triple_pruning(menu_ner, tuple_pairs_df, rest,
                                                                     review, ner_dict, store_in_db=True,
                                                                     graph_db=graph)
        triples_df = pd.concat([triples_df, tuple_pairs_prune])
        # if rev_index == 7:
        #     break
#     break

print(triples_df)


triples_df.to_csv("data/output/kg/input_data.txt-out1.csv", index=False)