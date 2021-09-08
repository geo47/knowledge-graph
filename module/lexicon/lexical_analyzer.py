import spacy


class SpacyLexicalAnalyzer:

    def __init__(self):

        self.lexicon_file_path = 'data/opinion_lexicon-en/'

        self.SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
        self.OBJECTS = ["dobj", "dative", "attr", "oprd"]
        self.ATTRS = ["acomp"]
        self.POSS_NOUN = ["NOUN", "PROPN", "X"]

        self.nlp_small = None
        self.nlp_large = None

    def load_spacy_models(self, model='small'):
        if model == 'small':
            self.nlp_small = spacy.load('en_core_web_sm')
        elif model == 'large':
            self.nlp_large = spacy.load('en_core_web_lg')
        else:
            self.nlp_small = spacy.load('en_core_web_sm')
            self.nlp_large = spacy.load('en_core_web_lg')

    def load_opinion_lexicon(self):
        # Load opinion lexicon
        neg_file = open(self.lexicon_file_path + "negative-words.txt", encoding="ISO-8859-1")
        pos_file = open(self.lexicon_file_path + "positive-words.txt", encoding="ISO-8859-1")
        neg = [line.strip() for line in neg_file.readlines()]
        pos = [line.strip() for line in pos_file.readlines()]
        opinion_words = neg + pos
        return opinion_words, pos, neg

    # opinion_words, pos, neg = load_opinion_lexicon()

    def get_compounds(self, tokens):
        compounds = []
        lefts = list(tokens.lefts)
        compounds.extend([tok for tok in lefts if tok.dep_ == 'compound'])
        return compounds

    def get_subject_compound(self, v):
        subs = []
        compounds = []
        for tok in v.lefts:
            if tok.dep_ in self.SUBJECTS and tok.pos_ != "DET":
                compounds = self.get_compounds(tok)
                compounds.extend([tok])
        if compounds:
            subs.extend(compounds)
        return subs

    def get_object_compound(self, v):
        objs = []
        compounds = []
        for tok in v.rights:
            if tok.dep_ in self.OBJECTS:
                compounds = self.get_compounds(tok)
                compounds.extend([tok])
        if compounds:
            objs.extend(compounds)
        return objs

    def get_attribute_compound(self, av):
        objs = []
        compounds = []
        for tok in av.rights:
            if tok.dep_ in self.ATTRS:
                compounds = self.get_compounds(tok)
                compounds.extend([tok])
        if compounds:
            objs.extend(compounds)
        return objs

    def get_prep_object_compound(self, p):
        p_objs = []
        compounds = []
        for tok in p.rights:
            if tok.dep_ == 'pobj':
                compounds = self.get_compounds(tok)
                compounds.extend([tok])
        if compounds:
            p_objs.extend(compounds)
        return p_objs

    def get_prep(self, p):
        prep = False
        for tok in p.rights:
            if tok.dep_ == 'prep':
                prep = tok
        return prep

    def get_lexical_triplets_pairs(self, text_doc, verbose=False):

        doc = self.nlp_small(text_doc)

        #     print([(ent.text, ent.label_) for ent in doc.ents])

        triplets = []

        verbs = [tok for tok in doc if tok.pos_ == "VERB"]

        # getting sub, verb, obj triples
        # Angela, visit, silver spoon restaurant
        for v in verbs:
            subs = self.get_subject_compound(v)
            objs = self.get_object_compound(v)

            if subs and objs:
                triplets.append([' '.join(str(i) for i in subs), str(v), ' '.join(str(i) for i in objs)])

        # getting sub, verb_prep, p_obj triples
        # Angela, visit_with, Angela's friends
        for v in verbs:
            subs = self.get_subject_compound(v)
            prep = self.get_prep(v)

            p_objs = False
            if prep:
                p_objs = self.get_prep_object_compound(prep)

            if subs and p_objs:
                triplets.append(
                    [' '.join(str(i) for i in subs), str(v) + '_' + str(prep), ' '.join(str(i) for i in p_objs)])

        # getting sub, possession, obj triples
        # Silver spoon restaurant, has, Chicken biryani
        subs, objs = (False, False)
        poss_nouns = [tok for tok in doc if tok.pos_ in self.POSS_NOUN]
        for n in poss_nouns:
            children = list(n.children)
            for child in children:
                if child.dep_ == 'poss' and child.pos_ in self.POSS_NOUN:
                    compounds = self.get_compounds(child)
                    compounds.extend([child])
                    subs = compounds

                    compounds = self.get_compounds(n)
                    compounds.extend([n])
                    objs = compounds

                    if subs and objs:
                        triplets.append([' '.join(str(i) for i in subs), 'has', ' '.join(str(i) for i in objs)])

        # getting sub, aux, complementary object triples
        # food, are, good
        subs, objs = (False, False)
        aux_verbs = [tok for tok in doc if tok.pos_ == "AUX"]
        for av in aux_verbs:
            subs = self.get_subject_compound(av)
            objs = self.get_attribute_compound(av)

            if subs and objs:
                triplets.append([' '.join(str(i) for i in subs), str(av), ' '.join(str(i) for i in objs)])

        #     pprint(triplets)
        return triplets

    def split_sentence(self, text):
        '''
        splits review into a list of sentences using spacy's sentence parser
        '''
        review = self.nlp_small(text)
        bag_sentence = []
        start = 0
        for token in review:
            if token.sent_start:
                bag_sentence.append(review[start:(token.i - 1)])
                start = token.i
            if token.i == len(review) - 1:
                bag_sentence.append(review[start:(token.i + 1)])
        return bag_sentence