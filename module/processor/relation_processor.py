import csv
import pandas as pd

from module.lexicon.lexical_analyzer import SpacyLexicalAnalyzer
from utils.spell_check import levenshtein_ratio_and_distance

# def triple_pruning(triples, ner_dict):
#     entity_set = set(ner_dict.keys())
#     final_triples = []
#
#     for row, col in triples.iterrows():
#         col['subject'] = col['subject'].strip()
#
#         # check if Named Entity in subject sentence fragment
#         # found_entity = False
#         # for named_entity in entity_set:
#         #     if named_entity in col['subject']:
#         #         col['subject'] = named_entity
#         #         found_entity = True
#         #
#         # if found_entity:
#         final_triples.append(('Node', col['subject'], col['relation'], 'Node', col['object']))
#
#     triple_df = pd.DataFrame(final_triples, columns=['Type1', 'Entity1', 'Relationship', 'Type2', 'Entity2']).drop_duplicates()
#     return triple_df


class TripletProcessor:

    def __init__(self):

        self.REL_IS = ["am", "be"]
        self.REL_HAVE = ["have", "had", "order", "try", "get"]

        self.kb_restaurant_dict = {}
        self.kb_restaurant_aspects_dict = {}
        self.kb_menu_dict = {}
        # self.kb_menu_aspects_dict = {}
        self.kb_general_dict = {}

        self.unique_to_node = False

        self.spacy_model_small = None
        self.opinion_words, self.pos, self.neg = ([], [], [])

    def init_lexical_analyzer(self):
        spacy_lexical_analyzer = SpacyLexicalAnalyzer()
        spacy_lexical_analyzer.load_spacy_models(model='small')
        self.spacy_model_small = spacy_lexical_analyzer.nlp_small
        self.opinion_words, self.pos, self.neg = spacy_lexical_analyzer.load_opinion_lexicon()

    def init_kb_dict(self, kb_restaurant_file, kb_restaurant_aspects_file, kb_menu_file):
        ''' read restaurant NER dictionary '''
        with open(kb_restaurant_file, mode='r') as infile:
            reader = csv.reader(infile)
            i = 0
            for rows in reader:
                if i == 0:
                    i += 1
                    continue
                self.kb_restaurant_dict[rows[0]] = rows[1]

        ''' read restaurant aspects NER dictionary '''
        with open(kb_restaurant_aspects_file, mode='r') as infile:
            reader = csv.reader(infile)
            i = 0
            for rows in reader:
                if i == 0:
                    i += 1
                    continue
                self.kb_restaurant_aspects_dict[rows[0]] = rows[1]

        ''' read menu NER dictionary '''
        with open(kb_menu_file, mode='r') as infile:
            reader = csv.reader(infile)
            i = 0
            for rows in reader:
                if i == 0:
                    i += 1
                    continue
                self.kb_menu_dict[rows[0]] = rows[1]

    def process_triple_pruning(self, menu_ner_model, triples, rest, review, ner_dict, store_in_db=False, graph_db=None):
        # entity_set = set(ner_dict.keys())
        final_triples = []
        menus_for_data = []
        aspects_for_data = []

        for row, col in triples.iterrows():
            sub_ent = ""
            obj_ent = ""
            # col['subject'] = col['subject'].strip().title()
            # col['object'] = col['object'].strip().title()
            col['subject'] = col['subject'].strip()
            col['object'] = col['object'].strip()
            # col['relation'] = col['relation'].upper()

            ''' Fixing Misspelling Menus '''
            valid_menu = False
            menu_subject = False
            for key, value in self.kb_menu_dict.items():
                if key.lower() == col['subject'].lower():
                    valid_menu = True
                    menu_subject = True

                    menus_for_data.append(col['subject'])
                    break

            if not valid_menu:
                for key, value in self.kb_menu_dict.items():
                    # Check if misspelled menu word
                    subject_ratio = levenshtein_ratio_and_distance(key.lower(), col['subject'].lower(),
                                                                   ratio_calc=True)
                    if subject_ratio > 0.85:
                        # Appending menu in the sentence, so that it could recognize the menu entity from the
                        # give sequence pattern
                        entity_word = menu_ner_model.extract_menu_ner_single(
                            "I like " + col['subject'].lower() + " from this restaurant.")
                        correction_word = menu_ner_model.extract_menu_ner_single(
                            "I like " + key.lower() + " from this restaurant.")

                        # key_id = "_".join(k for k in key.split())
                        if entity_word and correction_word:
                            print("updating subject [" + col['subject'] + "] with [" + key + "]")
                            review['text'] = review['text'].replace(col['subject'], key.title())
                            col['subject'] = key.title()
                            menu_subject = True

                            menus_for_data.append(col['subject'])
                            break

            valid_menu = False
            menu_object = False
            for key, value in self.kb_menu_dict.items():
                if key.lower() == col['object'].lower():
                    valid_menu = True
                    menu_object = True

                    menus_for_data.append(col['object'])
                    break

            if not valid_menu:
                for key, value in self.kb_menu_dict.items():
                    # Check if misspelled menu word
                    object_ratio = levenshtein_ratio_and_distance(key.lower(), col['object'].lower(), ratio_calc=True)
                    if object_ratio > 0.85:
                        # Appending menu in the sentence, so that it could recognize the menu entity from the
                        # give sequence pattern
                        entity_word = menu_ner_model.extract_menu_ner_single(
                            "I like " + col['object'].lower() + " from this restaurant.")
                        correction_word = menu_ner_model.extract_menu_ner_single(
                            "I like " + key.lower() + " from this restaurant.")

                        if entity_word and correction_word:
                            print("updating object [" + col['object'] + "] with [" + key + "]")
                            review['text'] = review['text'].replace(col['object'], key.title())
                            col['object'] = key.title()
                            menu_object = True

                            menus_for_data.append(col['object'])
                            break

            col['subject'] = col['subject'].strip().title()
            col['object'] = col['object'].strip().title()

            ''' check if subject is a valid entity '''
            valid_subject = False

            if menu_subject:
                menu_id = "_".join(k.title() for k in col['subject'].split(" "))
                col['subject'] = menu_id
                valid_subject = True

            restaurant_subject = False
            if not menu_subject:
                if rest['name'].lower() == col['subject'].lower():
                    col['subject'] = rest['rest_id']
                    restaurant_subject = True
                    valid_subject = True

            # if not menu_subject:
            #     for key, value in self.kb_restaurant_dict.items():
            #         if key.lower() == col['subject'].lower():
            #             col['subject'] = rest['rest_id']
            #             restaurant_subject = True
            #             valid_subject = True
            #             break

            #         menu_subject = False
            #         if not restaurant_subject:
            #             for key, value in kb_menu_dict.items():
            #                 if key.lower() == col['subject'].lower():
            #                     menu_id = "_".join(k for k in col['subject'].split())
            #                     col['subject'] = menu_id
            #                     menu_subject = True
            #                     valid_subject = True
            #                     break

            user_subject = False
            if not menu_subject and not restaurant_subject:
                if review['name'].lower() == col['subject'].lower():
                    col['subject'] = review['user_id']
                    user_subject = True
                    valid_subject = True

            # general_subject = False
            # if not user_subject:
            #     for key, value in ner_dict.items():
            #         if key.lower() == col['subject'].lower():
            #             general_subject = True
            #             valid_subject = True
            #             break

            res_aspect_subject = False
            if not menu_subject and not restaurant_subject and not user_subject:
                # Convert the words into base form of word
                doc = self.spacy_model_small(col['subject'])
                col['subject'] = " ".join(token.lemma_ for token in doc)

                for key, value in self.kb_restaurant_aspects_dict.items():
                    if key.lower() == col['subject'].lower():
                        aspect_id = "_".join(k.title() for k in col['subject'].split(" "))
                        col['subject'] = aspect_id
                        res_aspect_subject = True
                        valid_subject = True

                        aspects_for_data.append(col['subject'])
                        break

            # menu_aspect_subject = False
            # if not general_subject:
            #     for key, value in kb_menu_aspects_dict.items():
            #         if key.lower() == col['subject'].lower():
            #             menu_aspect_subject = True
            #             valid_subject = True
            #             break

            ''' check if object is a valid entity '''
            valid_object = False

            if menu_object:
                menu_id = "_".join(k.title() for k in col['object'].split(" "))
                col['object'] = menu_id
                valid_object = True

            restaurant_object = False
            if not menu_object:
                if rest['name'].lower() == col['object'].lower():
                    col['object'] = rest['rest_id']
                    restaurant_object = True
                    valid_object = True

            # restaurant_object = False
            # if not menu_object:
            #     for key, value in self.kb_restaurant_dict.items():
            #         if key.lower() == col['object'].lower():
            #             col['object'] = rest['rest_id']
            #             restaurant_object = True
            #             valid_object = True
            #             break

            #         menu_object = False
            #         if not restaurant_object:
            #             for key, value in kb_menu_dict.items():
            #                 if key.lower() == col['object'].lower():
            #                     menu_id = "_".join(k for k in col['object'].split())
            #                     col['object'] = menu_id
            #                     menu_object = True
            #                     valid_object = True
            #                     break

            user_object = False
            if not menu_object and not restaurant_object:
                if review['name'].lower() == col['object'].lower():
                    col['object'] = review['user_id']
                    user_object = True
                    valid_object = True

            general_object = False
            if not menu_object and not restaurant_object and not user_object:
                for key, value in ner_dict.items():
                    if key.lower() == col['object'].lower():
                        col['object'] = col['object'].title()
                        general_object = True
                        valid_object = True
                        break

            res_aspect_object = False
            if not menu_object and not restaurant_object and not user_object and not general_object:
                # Convert the words into base form of word
                doc = self.spacy_model_small(col['object'])
                col['object'] = " ".join(token.lemma_ for token in doc)

                for key, value in self.kb_restaurant_aspects_dict.items():
                    if key.lower() == col['object'].lower():
                        aspect_id = "_".join(k.title() for k in col['object'].split(" "))
                        col['object'] = aspect_id
                        res_aspect_object = True
                        valid_object = True

                        aspects_for_data.append(col['object'])
                        break

            attr_obj = False
            if not menu_object and not restaurant_object and not user_object and not general_object \
                    and not res_aspect_object:
                # Convert the words into base form of word
                doc = self.spacy_model_small(col['object'])
                col['object'] = " ".join(token.lemma_ for token in doc)

                # opinion_words, pos, neg = load_opinion_lexicon()
                if col['object'].lower() in self.opinion_words:
                    attr_id = "_".join(k.title() for k in col['object'].split(" "))
                    col['object'] = attr_id
                    attr_obj = True
                    valid_object = True

            # Todo Do your work HERE...

            # print(col['subject']+" : "+col['object']+" : "+str(valid_subject)+" : "+str(valid_object))
            if valid_subject and valid_object:

                # IF MENU:SUB
                if menu_subject: # Todo process relationship
                    menu_name = " ".join(token for token in col['subject'].split("_"))
                    sub_ent = menu_name
                    menu_id = col['subject']
                    if self.unique_to_node:
                        menu_id = rest['rest_id'] + "/" + col['subject']
                        col['subject'] = menu_id

                    final_triples.append((rest['name'], rest['rest_id'], 'HAS_MENU', sub_ent, col['subject']))

                    # Insert menu in Neo4j db
                    if store_in_db:
                        graph_db.create_menu(menu_id, menu_name)
                        graph_db.create_restaurant_has_menu_relation(rest['rest_id'], menu_id)

                # IF RESTAURANT:SUB and MENU:OBJ
                if restaurant_subject and menu_object:
                    col['relation'] = 'HAS_MENU'
                    sub_ent = rest['name']

                    menu_name = " ".join(token for token in col['object'].split("_"))
                    obj_ent = menu_name
                    menu_id = col['object']
                    if self.unique_to_node:
                        menu_id = col['subject'] + "/" + col['object']
                        col['object'] = menu_id

                    # Insert menu in Neo4j db
                    if store_in_db:
                        graph_db.create_menu(menu_id, menu_name)
                        graph_db.create_restaurant_has_menu_relation(col['subject'], menu_id)

                # IF User:SUB
                if user_subject and menu_object:
                    # Convert the relation word into base form of word
                    doc = self.spacy_model_small(col['relation'])
                    col['relation'] = str(" ".join(token.lemma_ for token in doc))

                    sub_ent = review['name']
                    # Make MENU object with rest-menu id
                    menu_name = " ".join(token for token in col['object'].split("_"))
                    obj_ent = menu_name
                    menu_id = col['object']
                    if self.unique_to_node:
                        menu_id = rest['rest_id'] + "/" + col['object']
                        col['object'] = menu_id

                    # If Menu is the object of user:sub relation add a rest_has_menu relation to the relation_triples
                    final_triples.append((rest['name'], rest['rest_id'], 'HAS_MENU', obj_ent, col['object']))

                    # Make relation with only ORDER type, ignore all relation predicate between User <-> Menu
                    col['relation'] = "ORDER"

                    # Insert menu in Neo4j db and create relation 'ORDER' with User
                    if store_in_db:
                        graph_db.create_menu(menu_id, menu_name)
                        graph_db.create_user_order_menu_relation(col['subject'], col['object'])
                        graph_db.create_restaurant_has_menu_relation(rest['rest_id'], col['object'])

                    # Make relation
                    '''if col['relation'] in self.REL_HAVE:
                        col['relation'] = "ORDER"

                        # Insert menu in Neo4j db and create relation 'ORDER' with User
                        if store_in_db:
                            graph_db.create_menu(menu_id, menu_name)
                            graph_db.create_user_order_menu_relation(col['subject'], col['object'])
                            graph_db.create_restaurant_has_menu_relation(rest['rest_id'], col['object'])
                    # User - LIKE -> Menu
                    else:
                        # Insert menu in Neo4j db and create open relation with User
                        if store_in_db:
                            graph_db.create_menu(menu_id, menu_name)
                            graph_db.create_user_menu_relation(col['subject'], col['object'], col['relation'].upper())
                            graph_db.create_restaurant_has_menu_relation(rest['rest_id'], col['object'])'''

                    # IF User:SUB and not Menu:OBJ - this is just for making (User - predicate - obj) for csv not db
                    if user_subject and not menu_object:
                        # Convert the relation word into base form of word
                        doc = self.spacy_model_small(col['relation'])
                        col['relation'] = str(" ".join(token.lemma_ for token in doc))

                        sub_ent = review['name']

                        _id = col['object']
                        if col['object'].startswith("b-"):
                            obj_ent = rest['name']
                        else:
                            # Make MENU object with rest-menu id
                            obj = " ".join(token for token in col['object'].split("_"))
                            obj_ent = obj
                            if self.unique_to_node:
                                _id = rest['rest_id'] + "/" + col['object']
                                col['object'] = _id

                # IF MENU:SUB and ATTR:OBJ
                if menu_subject and (attr_obj or (col['object'].lower() in ["spicy", "sweet"])):
                    # Convert the relation word into base form of word
                    doc = self.spacy_model_small(col['relation'])
                    col['relation'] = " ".join(token.lemma_ for token in doc)

                    # Make MENU object with rest-menu id
                    menu_name = " ".join(token for token in col['subject'].split("_"))
                    sub_ent = menu_name
                    if self.unique_to_node:
                        menu_name = " ".join(token for token in col['subject'].split("/")[1].split("_"))

                    # Make Attribute object
                    attr_name = " ".join(token for token in col['object'].split("_"))
                    obj_ent = attr_name
                    attr_id = col['object']
                    if self.unique_to_node:
                        attr_id = rest['rest_id'] + "/" + col['object']
                        col['object'] = attr_id

                    if store_in_db:
                        graph_db.create_attr(col['object'], attr_name)

                    # Make relation
                    if col['relation'] in self.REL_IS:
                        col['relation'] = "IS"

                        # Insert menu in Neo4j db and create relation 'IS' with attr
                        if store_in_db:
                            graph_db.create_menu(col['subject'], menu_name)
                            graph_db.create_menu_is_attr_relation(col['subject'], col['object'])

                    else:
                        # Insert aspect in Neo4j db and create open relation with Attribute
                        if store_in_db:
                            graph_db.create_menu(col['subject'], menu_name)
                            graph_db.create_menu_attr_relation(col['subject'], col['object'], col['relation'])

                    # add an extra relation from Menu attribute to Restaurant to show that the Menu attribute reflects
                    # to a particular Restaurant's Menu
                    if not self.unique_to_node:
                        final_triples.append((obj_ent, col['object'], 'MENU_ATTR_FOR', rest['name'], rest['rest_id']))
                        if store_in_db:
                            graph_db.create_attr_rest_relation(col['object'], rest['rest_id'], 'MENU_ATTR_FOR')

                # if res_aspect_object
                # IF RESTAURANT:SUB and ASPECT:OBJ
                if restaurant_subject and res_aspect_object:
                    col['relation'] = 'HAS_ASPECT'

                    sub_ent = rest['name']

                    aspect_name = " ".join(token for token in col['object'].split("_"))
                    obj_ent = aspect_name
                    aspect_id = col['object']
                    if self.unique_to_node:
                        aspect_id = col['subject'] + "/" + col['object']
                        col['object'] = aspect_id

                    # Insert menu in Neo4j db
                    if store_in_db:
                        graph_db.create_aspect(aspect_id, aspect_name)
                        graph_db.create_restaurant_has_aspect_relation(col['subject'], aspect_id)

                # IF RESTAURANT:SUB and Attr:OBJ
                if restaurant_subject and attr_obj:
                    # Convert the relation word into base form of word
                    doc = self.spacy_model_small(col['relation'])
                    col['relation'] = " ".join(token.lemma_ for token in doc)

                    sub_ent = rest['name']

                    # Make Attribute object
                    attr_name = " ".join(token for token in col['object'].split("_"))
                    obj_ent = attr_name
                    attr_id = col['object']
                    if self.unique_to_node:
                        attr_id = rest['rest_id'] + "/" + col['object']
                        col['object'] = attr_id

                    if store_in_db:
                        graph_db.create_attr(col['object'], attr_name)

                    # Make relation
                    if col['relation'] in self.REL_IS:
                        col['relation'] = "IS"
                        if store_in_db:
                            graph_db.create_restaurant_is_attr_relation(col['subject'], col['object'])
                    else:
                        if store_in_db:
                            graph_db.create_restaurant_attr_relation(col['subject'], col['object'], col['relation'])

                # IF Aspect:SUB and Attr:OBJ
                if res_aspect_subject and attr_obj:
                    # Convert the relation word into base form of word
                    doc = self.spacy_model_small(col['relation'])
                    col['relation'] = " ".join(token.lemma_ for token in doc)

                    # Make ASPECT object with rest-aspect id
                    aspect_name = " ".join(token for token in col['subject'].split("_"))
                    sub_ent = aspect_name
                    aspect_id = col['subject']
                    if self.unique_to_node:
                        aspect_id = rest['rest_id'] + "/" + col['subject']
                        col['subject'] = aspect_id

                    # Make Attribute object
                    attr_name = " ".join(token for token in col['object'].split("_"))
                    obj_ent = attr_name
                    attr_id = col['object']
                    if self.unique_to_node:
                        attr_id = rest['rest_id'] + "/" + col['object']
                        col['object'] = attr_id

                    if store_in_db:
                        graph_db.create_attr(col['object'], attr_name)

                    # If Aspect is subject add a rest_has_aspect relation to the relation_triples
                    final_triples.append((rest['name'], rest['rest_id'], 'HAS_ASPECT', sub_ent, col['subject']))

                    # Make relation
                    if col['relation'] in self.REL_IS:
                        col['relation'] = "IS"

                        # Insert menu in Neo4j db and create relation 'ORDER' with User
                        if store_in_db:
                            graph_db.create_aspect(aspect_id, aspect_name)
                            graph_db.create_restaurant_has_aspect_relation(rest['rest_id'], aspect_id)
                            graph_db.create_aspect_is_attr_relation(col['subject'], col['object'])

                    else:
                        # Insert aspect in Neo4j db and create open relation with Attribute
                        if store_in_db:
                            graph_db.create_aspect(aspect_id, aspect_name)
                            graph_db.create_restaurant_has_aspect_relation(rest['rest_id'], aspect_id)
                            graph_db.create_aspect_attr_relation(col['subject'], col['object'], col['relation'])

                    # add an extra relation from Aspect attribute to Restaurant to show that the Aspect
                    # attribute reflects to a particular Restaurant's Aspect
                    if not self.unique_to_node:
                        final_triples.append((obj_ent, col['object'], 'ASPECT_ATTR_FOR', rest['name'], rest['rest_id']))
                        if store_in_db:
                            graph_db.create_attr_rest_relation(col['object'], rest['rest_id'], 'ASPECT_ATTR_FOR')

                # making this condition filter for final triples
                if not (user_subject and attr_obj):
                    final_triples.append((sub_ent, col['subject'], col['relation'].upper(), obj_ent, col['object']))

        triple_df = pd.DataFrame(final_triples, columns=['sub_ent', 'subject', 'relation', 'obj_ent', 'object']) \
            .drop_duplicates()

        return triple_df, review, menus_for_data, aspects_for_data
