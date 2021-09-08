import glob
import argparse
import logging
import pickle
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


from IPython.core.display import display, HTML
from libs.gpr_pub import visualization


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

    doc = "Angela visit silver spoon. Angela usually visit this place every month with my friends. Their food " \
          "quality is amazing. Angela specially like chicken biryani. It is so " \
          "delicious. Their food as well as services are all good."
    # doc = "The cat sat on the mat. He quickly ran to the market. The dog jumped into the water. The author is " \
    #       "writing a book."
    # pos_pattern(doc)

    # pprint(pairs)

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ner', action='store_true', help="Resolve Co-reference")
    parser.add_argument('--ner_type', type=str, default="allen", help="Possible values (spacy|nltk|stanford|allen) "
                                                                      "Used only if --use_ner is true")
    parser.add_argument('--use_cr', action='store_true', default=False, help="Co-reference Resolution")
    parser.add_argument('--cr_type', type=str, default="allen", help="Possible values (spacy|stanford|allen) Used "
                                                                     "only if --use_cr is true")
    parser.add_argument('--use_re', action='store_true', default=False, help="Relation Extraction")
    parser.add_argument('--core_nlp_path', type=str,
                        default="./stanford-corenlp-4.2.2",
                        help="Path of Core NLP library, required when --use_cr is true and --cr_type is stanford")
    parser.add_argument('--verbose', action='store_true', default=True, help="Log")
    parser.add_argument('--text', type=str, help="Log")

    args = parser.parse_args()

    doc = args.text

    # Add css styles and js events to DOM, so that they are available to rendered html
    # display(HTML(open('libs/gpr_pub/visualization/highlight.css').read()))
    # display(HTML(open('libs/gpr_pub/visualization/highlight.js').read()))

    logger = logging.getLogger('main')
    logger.disabled = False

    verbose = False #args.verbose

    output_path = "./data/output/"
    ner_pickles_op = output_path + "ner/"
    cr_pickles_op = output_path + "cr/"

    # doc = "Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party in Las Vegas."
    # doc = "Angela usually visit silver spoon restaurant with my friends. Their food quality is amazing and Angela " \
    #       "personally like their chicken biryani a lot. Its so delicious. Their food as well as services are all good."

    # doc = "Angela usually visit silver spoon restaurant with Angela friends. silver spoon restaurant food quality " \
    #       "is amazing. Angela personally like silver spoon restaurant chicken biryani a lot. silver spoon " \
    #       "restaurant food as well as services are all good"

    # file_list = []
    # for f in glob.glob('./data/input/*'):
    #     file_list.append(f)
    #
    # for file in file_list:
    #     with open(file, "r") as f:
    #         lines = f.read().splitlines()
    #
    #     doc = ""
    #     for line in lines:
    #         doc += str(line + "\n")

    if verbose:
        print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
        print('\n\n')
        print('doc: '+doc)

    if args.use_ner:
        named_entities = {}
        if args.ner_type == 'nltk':
            named_entities = use_nltk_ner(doc, verbose)
        elif args.ner_type == 'spacy':
            named_entities = use_spacy_ner(doc, verbose)
        elif args.ner_type == 'stanford':
            named_entities = use_stanford_ner(doc, verbose)
        else:
            named_entities = use_allen_ner(doc, verbose)

        # html = visualization.render(named_entities, allen=True, jupyter=False)
        # display(html)

        op_pickle_filename = ner_pickles_op + "named_entity.pickle"
        with open(op_pickle_filename, "wb") as f:
            pickle.dump(named_entities, f)

        import json
        ner_filename = ner_pickles_op + "named_entity.txt"
        with open(ner_filename, "w") as f:
            f.write(json.dumps(named_entities))

    if args.use_cr:
        coref_resolved = ""
        if args.cr_type == 'spacy':
            # Todo spacy_cr is still in progress..
            spacy_cr = SpacyCR(verbose)
            spacy_cr.coreference_resolution(doc)
        elif args.cr_type == 'stanford':
            stanford_cr = StanfordCR(verbose)
            original_text, coref_resolved = stanford_cr.resolve_coreferences(doc)
        else:
            allen_cr = AllenCR(verbose)
            # working but not using, we use improved_replace_coref for Allen Coreference resolution
            # original_text, coref_resolved = allen_cr.coreference_resolution(doc)
            # data = allen_cr.coreference_resolution(doc)
            # html = visualization.render(data, allen=True, jupyter=False)
            # display(html)

            predictor, nlp = allen_cr.load_models()
            clusters = predictor.predict(doc)['clusters']
            nlp_doc = nlp(doc)
            # allen_cr.print_comparison(allen_cr.original_replace_corefs(nlp_doc, clusters),
            #                            allen_cr.improved_replace_corefs(nlp_doc, clusters))
            # coref_resolved = allen_cr.original_replace_corefs(nlp_doc, clusters)
            coref_resolved = allen_cr.improved_replace_corefs(nlp_doc, clusters)

        # doc = coref_resolved
        op_pickle_filename = cr_pickles_op + "cr_resolution.pickle"
        with open(op_pickle_filename, "wb") as f:
            pickle.dump(coref_resolved, f)

        coreference_file = cr_pickles_op + "cr_resolution.txt"
        with open(coreference_file, "w") as f:
            f.write(coref_resolved)

        print('\n')
        print('\n')
        print('################ Original Text #################')
        print(doc)
        print('################################################')

        print('################ Coreference Resolved #################')
        print(coref_resolved)
        print('################################################')

    if args.use_re:
        allen_re = AllanRE(verbose)
        # allen_re.extract_relations(doc)

        with StanfordOpenIE() as client:
            # text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
            # text = "Angela usually visit silver spoon restaurant with Angela's friends. silver spoon " \
            #        "restaurant's food quality is amazing and Angela like silver spoon restaurant's " \
            #        "chicken biryani a lot. silver spoon restaurant's chicken biryani so delicious. silver spoon " \
            #        "restaurant's food as well as services are all good."
            text = doc
            print('Text: %s.' % text)

            sentences = split_sentence(text)

            triples = []
            for sentence in sentences:
                print("processing sentence: ")
                print(sentence)

                for triple in client.annotate(str(sentence)):
                    tri = []

                    print('|-', triple)
                    tri.append(triple.get("subject"))
                    tri.append(triple.get("relation"))
                    tri.append(triple.get("object"))
                    triples.append(tri)

            print(triples)

            tuple_pairs = get_lexical_triplets_pairs(text)
            pairs = list(set(tuple(sub) for sub in tuple_pairs))
            # pairs.append(['silver spoon restaurant','like','chicken biryani'])
            # pairs.append(['services', 'are', 'good'])
            triples.extend(pairs)



            df = pd.DataFrame(pairs, columns=['subject', 'relation', 'object'])
            print(df)
            df.to_csv("data/output/kg/input_data.txt-out.csv", index=False)
            from create_kg import draw_kg
            draw_kg(df)

            graph = list()
            graph.append('digraph {')
            for er in pairs:
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

