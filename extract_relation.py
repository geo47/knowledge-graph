from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import allennlp_models.tagging


class AllanRE:

    def __init__(self, verbose):
        print("[AllanRE] Initializing...")

        self.verbose = verbose

    def extract_relations(self, doc):
        model_url = 'https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz'
        if self.verbose:
            print("[AllanRE] Loading model from: "+model_url)

        predictor = Predictor.from_path(model_url)
        if self.verbose:
            print("[AllanRE] Model loaded successfully...")

        prediction = predictor.predict(sentence=doc)
        if self.verbose:
            print("[AllanRE]")
            print(prediction)

    def parse_results(self, results):
        relation_list = results['verbs']

        for relation in relation_list:
            rel = relation['description']

            # if rel

# # text = "In December, John decided to join the party.."
# # text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
# # text = "Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party in Las Vegas."
# # text = "Angela orders Chicken Biryani. She likes it a lot. It was so delicious. delicious means that was so amazing."
# text = "Pai never disappoints! Karina love their Thai green curry as well Masaman curry with rice. As a pescatarian Karina have plenty options. The food is always fresh and delicious!"
# model_url = 'https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz'
# predictor = Predictor.from_path(model_url)
#
# print(text)
# prediction = predictor.predict(sentence=text)
# print(prediction)
#
# # clusters = prediction['clusters']
# # words = prediction['document']
#
# # print(clusters)
# # print(words)