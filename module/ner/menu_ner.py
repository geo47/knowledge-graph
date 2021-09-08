
import csv

from transformers import BertModel, BertForTokenClassification, BertTokenizer
import torch


class MenuNER:

    def __init__(self):
        self.menu_file = r"data/kb/entities/menu.csv"
        self.entity_headers = ['Name', 'Label']

        self.model = BertForTokenClassification.from_pretrained(
            "/home/muzamil/Projects/Python/ML/NLP/NER/BERT-NER/out_ner/")
        self.tokenizer = BertTokenizer.from_pretrained(
            "/home/muzamil/Projects/Python/ML/NLP/NER/MenuNER/model/FoodieBERT/cased_L-12_H-768_A-12")

    def extract_menu_ner(self, restaurants, write_results=False):

        menu_ner = dict()
        for restaurant in restaurants['restaurants']:
            for review in restaurant['reviews']:
                sequence = review['text']
                predict = self._extract_menu_ner(self.model, self.tokenizer, sequence)
                menu_ner.update(predict)
        #                 print(predict)

        #         print(menu_ner)
        if write_results:
            # opening the csv file in 'w' mode
            open_menu_extracted_file = open(self.menu_file, 'w')
            writer = csv.DictWriter(open_menu_extracted_file, fieldnames=self.entity_headers)

            writer.writeheader()
            for key, value in menu_ner.items():
                writer.writerow({self.entity_headers[0]: key,
                                 self.entity_headers[1]: value})

    def extract_menu_ner_single(self, sequence):
        menu_ner = dict()
        predict = self._extract_menu_ner(self.model, self.tokenizer, sequence)
        menu_ner.update(predict)
        return predict

    @staticmethod
    def _extract_menu_ner(model, tokenizer, sequence):

        label_list = [
            "O",  # Outside of a named entity
            "B-MENU",  # Beginning of a menu entity
            "I-MENU",  # menu entity
        ]

        # Bit of a hack to get the tokens with the special tokens
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence, max_length=512, truncation=True)))
        inputs = tokenizer.encode(sequence, return_tensors="pt")

        predict = {}
        if inputs.size()[1] > 512:
            return predict

        outputs = model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)

        full_token = ''

        for token, prediction in zip(tokens, predictions[0].tolist()):
            if token != '[CLS]' and token != '[SEP]':
                if prediction > 3:
                    continue
                if label_list[prediction - 1] in ["B-MENU", "I-MENU"]:
                    if token.startswith('##'):
                        full_token = full_token + token.replace("##", "")
                    else:
                        if full_token:
                            full_token = full_token + " " + token
                        else:
                            full_token = token
                elif full_token:
                    if token.startswith('##'):
                        full_token = full_token + token.replace("##", "")
                    else:
                        predict[full_token] = "MENU"
                        full_token = ''

        # Make first letter capitan and all small case for MenuNER
        # predict = dict((key.title(), value) for (key, value) in predict.items())
        predict = dict((key, value) for (key, value) in predict.items())
        return predict
