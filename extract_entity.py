from pprint import pprint

import nltk
from stanza.server import CoreNLPClient
from allennlp.predictors.predictor import Predictor

import en_core_web_sm


class NltkNER:

    def __init__(self):
        print("[NltkNER] Using NltkNER")

    def ner(self, doc):
        pos_tagged = self.assign_pos_tags(doc)
        # chunks = self.split_into_chunks(pos_tagged)
        result = []
        for sent in pos_tagged:
            result.append(nltk.ne_chunk(sent))
        return result

    def assign_pos_tags(self, doc):
        sentences = nltk.sent_tokenize(doc)
        words = [nltk.word_tokenize(sent) for sent in sentences]
        pos_tagged = [nltk.pos_tag(word) for word in words]
        return pos_tagged

    # Chunk results are not good
    def split_into_chunks(self, sentences):
        # This rule says that an NP chunk should be formed whenever the chunker finds an optional determiner (DT)
        # or possessive pronoun (PRP$) followed by any number of adjectives (JJ/JJR/JJS) and then any number of
        # nouns (NN/NNS/NNP/NNPS) {dictator/NN Kim/NNP Jong/NNP Un/NNP}. Using this grammar, we create a chunk parser.
        grammar = "NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}"
        cp = nltk.RegexpParser(grammar)
        chunks = []
        for sent in sentences:
            chunks.append(cp.parse(sent))
        return chunks

    def ner_tree_to_dict(self, tree_list):

        tree_dict = dict()

        for tree in tree_list:
            for st in tree:
                # not everything gets a NE tag, so we can ignore untagged tokens
                # which are stored in tuples
                if isinstance(st, nltk.Tree):
                    tree_dict[' '.join([tup[0] for tup in st])] = st.label()

        return tree_dict

    def display(self, ner):
        print("[NltkNER]")
        pprint(ner)
        # for leaves in ner:
        #     print(leaves)
        #     leaves.draw()
        print("\n")


class SpacyNER:

    def __init__(self):
        print("[SpacyNER] Using SpacyNER")

    def ner(self, doc):
        nlp = en_core_web_sm.load()
        doc = nlp(doc)
        return [(X.text, X.label_) for X in doc.ents]

    def ner_to_dict(self, ner):
        """
        Expects ner of the form list of tuples
        """
        ner_dict = {}
        for tup in ner:
            ner_dict[tup[0]] = tup[1]
        return ner_dict

    def display(self, ner):
        print("[SpacyNER]")
        print(ner)


class StanfordNER:

    def __init__(self):
        print("[StanfordNER] Using StanfordNER; this may take some time...")

        import os
        os.environ["CORENLP_HOME"] = "/home/muzamil/Projects/Python/ML/NLP/KG/knowledge-graph/stanford-corenlp-4.2.2"

    def ner(self, doc):
        # set up the client

        import stanza
        nlp = stanza.Pipeline('en')

        with CoreNLPClient(annotators=['ner'], file='./data/ner_rules.txt', timeout=30000, memory='16G') as client:
            # submit the request to the server
            ann = client.annotate(doc)

            sentences = ann.sentence
            result = []
            for sentence in sentences:
                for token in sentence.token:
                    if token.ner != "O":
                        tuple = (token.value, token.ner)
                        result.append(tuple)
            return result

    def ner_to_dict(self, ner):
        """
        Expects ner of the form list of tuples
        """
        ner_dict = {}
        for tup in ner:
            ner_dict[tup[0]] = tup[1]
        return ner_dict

    def display(self, ner):
        print("[StanfordNER]")
        print(ner)


class AllenNER:

    def __init__(self):
        print("[AllenNER] Using AllenNER; this may take some time...")

    def ner(self, doc):
        model_url = "https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2021-02-11.tar.gz"
        predictor = Predictor.from_path(model_url)
        prediction = predictor.predict(sentence=doc)
        return prediction

    def ner_to_dict(self, result):

        ner_dict = {}
        words = result["words"]
        tags = result["tags"]
        for i in range(0, len(tags)):
            if tags[i] != 'O':
                ner_dict[words[i]] = tags[i]
        return ner_dict

    def display(self, ner):
        print("[AllenNER]")
        print(ner)


def use_nltk_ner(doc, verbose):
    nltk_ner = NltkNER()
    named_entities = nltk_ner.ner(doc)
    named_entities = nltk_ner.ner_tree_to_dict(named_entities)
    if verbose:
        nltk_ner.display(named_entities)
    return named_entities


def use_spacy_ner(doc, verbose):
    spacy_ner = SpacyNER()
    named_entities = spacy_ner.ner(doc)
    named_entities = spacy_ner.ner_to_dict(named_entities)
    if verbose:
        spacy_ner.display(named_entities)
    return named_entities


def use_stanford_ner(doc, verbose):
    stanford_ner = StanfordNER()
    named_entities = stanford_ner.ner(doc)
    named_entities = stanford_ner.ner_to_dict(named_entities)
    if verbose:
        stanford_ner.display(named_entities)
    return named_entities


def use_allen_ner(doc, verbose):
    allen_ner = AllenNER()
    named_entities = allen_ner.ner(doc)
    named_entities = allen_ner.ner_to_dict(named_entities)
    if verbose:
        allen_ner.display(named_entities)
    return named_entities




# def main():

    # doc = "Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party in Las Vegas."
    # doc = "Pai never disappoints! Karina love their Thai green curry as well Masaman curry with rice. As a pescatarian Karina have plenty options. The food is always fresh and delicious!"

    # print("Document: \n", doc)
    # print("\n")

    # use_nltk_ner(doc)
    # use_spacy_ner(doc)
    # use_stanford_ner(doc)
    # use_allen_ner(doc, True)





# main()