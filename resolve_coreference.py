import json
from typing import List
from spacy.tokens import Doc, Span

from stanza.server import CoreNLPClient
import spacy
import neuralcoref
from allennlp.predictors.predictor import Predictor

from utils.intersection_strategies.intersection_strategy import IntersectionStrategy


class StanfordCR:

    def __init__(self, verbose):

        self.verbose = verbose

        print("[StanfordCR] Initializing...")

        import os
        os.environ["CORENLP_HOME"] = "/home/muzamil/Projects/Python/ML/NLP/KG/knowledge-graph/libs/stanford-corenlp-4.2.2"

        self.stanford_core_nlp_path = 'libs/stanford-corenlp-4.2.2'

    def resolve_coreferences(self, doc):
        corefs = self.generate_coreferences(doc)
        # coref.unpickle()
        # result = coref_obj.resolve_coreferences(corefs, doc, ner, verbose)
        # return result
        return doc, corefs

    def generate_coreferences(self, doc):
        # set up the client
        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'],
                           timeout=30000, memory='16G') as client:

            # submit the request to the server and get json output
            ann = client.annotate(doc, output_format='json')
            ann2 = client.annotate(doc)

            # parse json
            ann_str = json.dumps(ann)
            result = json.loads(ann_str)

            sentences = ann2.sentence
            coref_resolver = []
            for sentence in sentences:
                sent_list = []
                for token in sentence.token:
                    sent_list.append(token.word)
                coref_resolver.append(sent_list)

            for num, mentions in list(result['corefs'].items()):

                i = 0
                head_mention = ''
                for mention in mentions:
                    i += 1
                    if i == 1:
                        head_mention = mention["text"]
                        continue
                    coref_resolver[mention["sentNum"]-1][mention["startIndex"]-1] = head_mention
                    # if self.verbose:
                    #     print("[StanfordCR]")
                    #     print(mention)

            if self.verbose:
                print("[StanfordCR] original_text: "+doc)

            coreference_resolved = ''
            for sent in coref_resolver:
                coreference_resolved += ' '.join(tok for tok in sent)
                coreference_resolved += ' '

            if self.verbose:
                print("[StanfordCR] coreference_resolved: " + coreference_resolved)
            return doc, coreference_resolved


class SpacyCR:

    def __init__(self, verbose):
        self.verbose = verbose

        print("[SpacyCR] Initializing...")

    def coreference_resolution(self, doc):

        nlp = spacy.load('en')
        neuralcoref.add_to_pipe(nlp)

        spacy_doc = nlp(doc)

        if self.verbose:
            print(spacy_doc._.has_coref)
            print(spacy_doc._.coref_clusters)


class AllenCR:

    '''
        Reference: https://github.com/NeuroSYS-pl/coreference-resolution/blob/main/improvements_to_allennlp_cr.ipynb
    '''
    def __init__(self, verbose):
        self.verbose = verbose

        print("[AllenCR] Initializing...")

    def load_models(self):
        model_url = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'
        if self.verbose:
            print("[AllenCR] Loading model from: " + model_url)
        predictor = Predictor.from_path(model_url)
        if self.verbose:
            print("[AllenCR] Model loaded successfully...")

        nlp = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(nlp)
        return predictor, nlp

    def get_cluster_head_idx(self, doc, cluster):
        noun_indices = IntersectionStrategy.get_span_noun_indices(doc, cluster)
        return noun_indices[0] if noun_indices else 0

    def print_clusters(self, doc, clusters):
        def get_span_words(span, allen_document):
            return ' '.join(allen_document[span[0]:span[1] + 1])

        allen_document, clusters = [t.text for t in doc], clusters
        for cluster in clusters:
            cluster_head_idx = self.get_cluster_head_idx(doc, cluster)
            if cluster_head_idx >= 0:
                cluster_head = cluster[cluster_head_idx]
                print(get_span_words(cluster_head, allen_document) + ' - ', end='')
                print('[', end='')
                for i, span in enumerate(cluster):
                    print(get_span_words(span, allen_document) + ("; " if i + 1 < len(cluster) else ""), end='')
                print(']')

    def print_comparison(self, resolved_original_text, resolved_improved_text):
        print(f"~~~ AllenNLP original replace_corefs ~~~\n{resolved_original_text}")
        print(f"\n~~~ Our improved replace_corefs ~~~\n{resolved_improved_text}")

    def coreference_resolution(self, doc):

        predictor, nlp = self.load_models()

        prediction = predictor.predict(document=doc)
        if self.verbose:
            print("[AllenCR]")
            print(prediction)

        # return prediction

        clusters = prediction['clusters']
        words = prediction['document']
        new_words = words
        del_len = 0
        for cluster in clusters:
            first_mention = 0
            head_mention = ""
            for mention in cluster:
                first_mention += 1
                if first_mention == 1:
                    head_mention = new_words[mention[0]:mention[1] + 1]
                    continue

                span = (mention[1] + 1) - mention[0]
                new_words[mention[0]] = ' '.join(token for token in head_mention)
                span = span - 1

                if span > 0:
                    del new_words[(mention[0] + 1):((mention[1] + 1) - del_len)]

                del_len += span

        original_text = " ".join(token for token in words)
        coreference_resolved = " ".join(token for token in new_words)
        if self.verbose:
            print("[AllenCR] ")
            print(new_words)
            print("[AllenCR] original_text: "+original_text)
            print("[AllenCR] coreference_resolved: "+coreference_resolved)

        return original_text, coreference_resolved

    def core_logic_part(self, document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
        final_token = document[coref[1]]
        if final_token.tag_ in ["PRP$", "POS"]:
            resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
        else:
            resolved[coref[0]] = mention_span.text + final_token.whitespace_
        for i in range(coref[0] + 1, coref[1] + 1):
            resolved[i] = ""
        return resolved

    def original_replace_corefs(self, document: Doc, clusters: List[List[List[int]]]) -> str:
        resolved = list(tok.text_with_ws for tok in document)

        for cluster in clusters:
            mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
            mention_span = document[mention_start:mention_end]

            for coref in cluster[1:]:
                self.core_logic_part(document, coref, resolved, mention_span)

        return "".join(resolved)

    def get_span_noun_indices(self, doc: Doc, cluster: List[List[int]]) -> List[int]:
        spans = [doc[span[0]:span[1] + 1] for span in cluster]
        spans_pos = [[token.pos_ for token in span] for span in spans]
        span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
                             if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
        return span_noun_indices

    def is_containing_other_spans(self, span: List[int], all_spans: List[List[int]]):
        return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

    def get_cluster_head(self, doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
        head_idx = noun_indices[0]
        head_start, head_end = cluster[head_idx]
        head_span = doc[head_start:head_end + 1]
        return head_span, [head_start, head_end]

    def improved_replace_corefs(self, document, clusters):
        resolved = list(tok.text_with_ws for tok in document)
        all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

        for cluster in clusters:
            noun_indices = self.get_span_noun_indices(document, cluster)

            if noun_indices:
                mention_span, mention = self.get_cluster_head(document, cluster, noun_indices)

                for coref in cluster:
                    if coref != mention and not self.is_containing_other_spans(coref, all_spans):
                        self.core_logic_part(document, coref, resolved, mention_span)

        return "".join(resolved)

def main():

    # doc = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
    # doc = "Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party in " \
    #       "Las Vegas."
    # doc = "Angela orders Chicken Biryani. She likes it a lot. It was so delicious. delicious means that was " \
    #       "so amazing."
    doc = "Angela usually visit silver spoon restaurant with her friends . Their food quality is amazing and " \
          "Angela personally like their chicken biryani a lot . its so delicious . The food as well as services " \
          "are all good ."

    # doc = "Chesney went downstairs to his locker and came back with the five chairs. Nudging open the apartment " \
    #       "door, he was surprised to see a little blonde girl in pinafore and ankle socks standing beside the " \
    #       "table. 'Are you lost?' he said. 'I just got one question,' she said. Actually, the voice asking the " \
    #       "question came not from girl but from the fanged mouth of the rubyred snake that uncoiled itself where a " \
    #       "tongue would have been if this had really been a little girl instead of another demon. He put down " \
    #       "the chairs."

    # doc = "We want to take our code and create a game. Let's remind ourselves how to do that."
    # doc = '"He is a great actor!", he said about John Travolta.'
    # doc = "Anna likes Tom. Tom is Anna's brother. Her brother is tall."
    import logging
    logging.basicConfig(level=logging.INFO)

    # nlp = spacy.load('en')
    # neuralcoref.add_to_pipe(nlp)
    #
    # # spacy_doc = nlp(u'My sister has a dog. She loves him.')
    # spacy_doc = nlp(doc)
    # print(spacy_doc._.has_coref)
    # print(spacy_doc._.coref_clusters)

    # StanfordNLP
    # stanford_cr = StanfordCR(True)
    # stanford_cr.resolve_coreferences(doc)

    # AllanNLP
    allen_nlp = AllenCR(True)
    # allen_nlp.coreference_resolution(doc)


    # prediction = predictor.predict(document=doc)
    # print("[AllenCR]")
    # print(prediction)

    predictor, nlp = allen_nlp.load_models()

    clusters = predictor.predict(doc)['clusters']
    nlp_doc = nlp(doc)

    allen_nlp.print_comparison(allen_nlp.original_replace_corefs(nlp_doc, clusters),
                               allen_nlp.improved_replace_corefs(nlp_doc, clusters))


# main()



