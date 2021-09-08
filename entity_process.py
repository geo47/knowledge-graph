import csv
import pandas as pd

from sentence_transformers import SentenceTransformer, util

from utils.spell_check import levenshtein_ratio_and_distance

# facts_file = r"data/output/kg/input_data.txt-out.csv"
# lexicon_file_path = 'data/opinion_lexicon-en/'
#
# ''' read restaurant knowledgebase data '''
# knowledge_base_entities = r"data/kb/entities/"
# kb_restaurant_file = knowledge_base_entities + "restaurant.csv"
# kb_restaurant_aspects_file = knowledge_base_entities + "restaurant_aspects.csv"
# kb_menu_file = knowledge_base_entities + "menu.csv"
# # kb_menu_aspects_file = knowledge_base_entities + "menu_attrs.csv"
# kb_general_file = knowledge_base_entities + "general.csv"
#
# entity_headers = ['Name', 'Label']
#
# ''' make dataframes for kb '''
# kb_restaurant_df = pd.read_csv(kb_restaurant_file, header=0, names=entity_headers)
# kb_restaurant_aspects_df = pd.read_csv(kb_restaurant_aspects_file, header=0, names=entity_headers)
# kb_menu_df = pd.read_csv(kb_menu_file, header=0, names=entity_headers)
# # kb_menu_aspects_df = pd.read_csv(kb_menu_aspects_file, header=0, names=entity_headers)
# # kb_general_df = pd.read_csv(kb_general_file, header=0, names=['Name', 'Label'])
#
# kb_restaurant_dict = {}
# kb_restaurant_aspects_dict = {}
# kb_menu_dict = {}
# # kb_menu_aspects_dict = {}
# kb_general_dict = {}
#
# ''' read restaurant NER dictionary '''
# with open(kb_restaurant_file, mode='r') as infile:
#     reader = csv.reader(infile)
#     i = 0
#     for rows in reader:
#         if i == 0:
#             i += 1
#             continue
#         kb_restaurant_dict[rows[0]] = rows[1]
#
# ''' read restaurant aspects NER dictionary '''
# with open(kb_restaurant_aspects_file, mode='r') as infile:
#     reader = csv.reader(infile)
#     i = 0
#     for rows in reader:
#         if i == 0:
#             i += 1
#             continue
#         kb_restaurant_aspects_dict[rows[0]] = rows[1]
#
# ''' read menu NER dictionary '''
# with open(kb_menu_file, mode='r') as infile:
#     reader = csv.reader(infile)
#     i = 0
#     for rows in reader:
#         if i == 0:
#             i += 1
#             continue
#         kb_menu_dict[rows[0]] = rows[1]
#
# ''' read menu aspects NER dictionary '''
# # with open(kb_menu_aspects_file, mode='r') as infile:
# #     reader = csv.reader(infile)
# #     i = 0
# #     for rows in reader:
# #         if i == 0:
# #             i += 1
# #             continue
# #         kb_menu_aspects_dict[rows[0]] = rows[1]
#
# with open(kb_general_file, mode='r') as infile:
#     reader = csv.reader(infile)
#     i = 0
#     for rows in reader:
#         if i == 0:
#             i += 1
#             continue
#         kb_general_dict[rows[0]] = rows[1]
#
# print(kb_restaurant_dict)
# print(kb_restaurant_aspects_dict)
# print(kb_menu_dict)
# # print(kb_menu_aspects_dict)
# print(kb_general_dict)
#
# facts_df = pd.read_csv(facts_file)
# print(facts_df)


def load_opinion_lexicon():
    # Load opinion lexicon
    neg_file = open(lexicon_file_path + "negative-words.txt", encoding="ISO-8859-1")
    pos_file = open(lexicon_file_path + "positive-words.txt", encoding="ISO-8859-1")
    neg = [line.strip() for line in neg_file.readlines()]
    pos = [line.strip() for line in pos_file.readlines()]
    opinion_words = neg + pos
    return opinion_words, pos, neg

def process_triple_pruning(triples):
    # entity_set = set(ner_dict.keys())
    final_triples = []

    for row, col in triples.iterrows():
        col['subject'] = col['subject'].strip()
        col['object']  = col['object'].strip()

        for key, value in kb_menu_dict.items():
            subject_ratio = levenshtein_ratio_and_distance(key.lower(), col['subject'].lower(), ratio_calc=True)
            object_ratio  = levenshtein_ratio_and_distance(key.lower(), col['object'].lower(), ratio_calc=True)
            if subject_ratio > 0.85:
                print("updating subject ["+col['subject']+"] with ["+key+"]")
                col['subject'] = key

            if object_ratio > 0.85:
                print("updating object [" + col['object'] + "] with [" + key + "]")
                col['object'] = key

        ''' check if subject is a valid entity '''
        valid_subject = False
        restaurant_subject = False
        for key, value in kb_restaurant_dict.items():
            if key.lower() == col['subject'].lower():
                restaurant_subject = True
                valid_subject = True
                break

        menu_subject = False
        if not restaurant_subject:
            for key, value in kb_menu_dict.items():
                if key.lower() == col['subject'].lower():
                    menu_subject = True
                    valid_subject = True
                    break

        general_subject = False
        if not menu_subject:
            for key, value in kb_general_dict.items():
                if key.lower() == col['subject'].lower():
                    general_subject = True
                    valid_subject = True
                    break

        res_aspect_subject = False
        if not general_subject:
            for key, value in kb_restaurant_aspects_dict.items():
                if key.lower() == col['subject'].lower():
                    res_aspect_subject = True
                    valid_subject = True
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
        restaurant_object = False
        for key, value in kb_restaurant_dict.items():
            if key.lower() == col['object'].lower():
                restaurant_object = True
                valid_object = True
                break

        menu_object = False
        if not restaurant_object:
            for key, value in kb_menu_dict.items():
                if key.lower() == col['object'].lower():
                    menu_object = True
                    valid_object = True
                    break

        general_object = False
        if not menu_object:
            for key, value in kb_general_dict.items():
                if key.lower() == col['object'].lower():
                    general_object = True
                    valid_object = True
                    break

        res_aspect_object = False
        if not general_object:
            for key, value in kb_restaurant_aspects_dict.items():
                if key.lower() == col['object'].lower():
                    res_aspect_object = True
                    valid_object = True
                    break

        attr_obj = False
        if not res_aspect_object:
            opinion_words, pos, neg = load_opinion_lexicon()
            if col['object'].lower() in opinion_words:
                attr_obj = True
                valid_object = True

        if valid_subject and valid_object:
            if menu_subject:
                final_triples.append(('Node', 'Silver spoon restaurant', 'has_menu', 'Node', col['subject']))
            if restaurant_subject and menu_object:
                col['relation'] = 'has_menu'
            if res_aspect_subject and (res_aspect_object or attr_obj):
                col['relation'] = 'is'
            final_triples.append(('Node', col['subject'], col['relation'], 'Node', col['object']))

    triple_df = pd.DataFrame(final_triples, columns=['Type1', 'Entity1', 'Relationship', 'Type2', 'Entity2'])\
        .drop_duplicates()
    return triple_df


def process_entity_linking(triple_df, confidence_threshold):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    triple_df = linker(triple_df=triple_df, head_name='Entity2', tail_name='Entity2', model=model,
                       confidence_threshold=confidence_threshold)
    triple_df = linker(triple_df=triple_df, head_name='Entity1', tail_name='Entity1', model=model,
                       confidence_threshold=confidence_threshold)
    triple_df = linker(triple_df=triple_df, head_name='Entity1', tail_name='Entity2', model=model,
                       confidence_threshold=confidence_threshold)

    triple_df = triple_df.drop_duplicates()
    return triple_df


def linker(triple_df, head_name, tail_name, model, confidence_threshold):
    index = 1
    for _, col1 in triple_df.iterrows():
        head = col1[head_name]
        embedding1 = model.encode(head, convert_to_tensor=True)

        for _, col2 in triple_df.iterrows():
            tail = col2[tail_name]
            if head == tail:
                continue

            embedding2 = model.encode(tail, convert_to_tensor=True)
            confidence = util.pytorch_cos_sim(embedding1, embedding2)[0][0]

            if confidence > confidence_threshold:  # 85% seems to work pretty well
                # Perform logic for linking
                new_tail = tail if len(tail) < len(head) else head

                col1[head_name] = new_tail
                col2[tail_name] = new_tail

                # print("Sentence 1:", head)
                # print("Sentence 2:", tail)
                # print("Similarity:", confidence)
                # print('Processed {}\n'.format(index))
                index += 1

    return triple_df


# print(facts_df.head(300).to_string())
# triple_df = process_triple_pruning(triples=facts_df)
# print(triple_df.head(300).to_string())
# triple_df = process_entity_linking(triple_df=triple_df, confidence_threshold=0.75)
# print(triple_df.head(300).to_string())