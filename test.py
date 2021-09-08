# Using spaCy & NLP to create variations of "those generously buttered noodles"
# See here: https://twitter.com/ArielDumas/status/1086294656957272065
#
# Disclaimer 1: This is a quick, simplified example focusing on one particular
# sentence. There are obviously many more different constructions and
# different types of dependencies you want to cover. Some aspects also become
# significantly more difficult if you're working with, say, German instead of
# English.
#
# Disclaimer 2: Creating spam comments is a very bad use case for NLP and
# something I / we do not endorse. The point of this little script is to show
# how to use spaCy to analyse syntax and other fund things you can do with the
# dependency parse. So feel free to use this in your projects, but please don't
# build spam bots ;)

# Prerequisites:
# pip install spacy
# python -m spacy download en_core_web_sm
import spacy

# To process the text, we're using the small English model, which was trained
# on a corpus of general-purpose news and web text. See here for details:
# https://spacy.io/models/en#en_core_web_sm
# from textpipeliner import SequencePipe, NamedEntityFilterPipe, FindTokensPipe, NamedEntityExtractorPipe, AnyPipe, \
#     AggregatePipe, PipelineEngine, Context

nlp = spacy.load("en_core_web_sm")

# Here's our original text that we want to rephrase.
# text = "These generously buttered noodles, sprinkled with just a quarter cup of parsley for color and freshness, are the perfect blank canvas for practically any stew or braise."
text = "I Ordered Chicken Biryani."
# This is the template we want to fill based on the original text. The subject
# of the original sentence becomes an object attached to "love".
template = "Couldn't agree more, but I would add that I sincerely love {subject}, because {pronoun} {verb} {obj}."

# Calling the nlp object on a string of text returns a processed Doc object,
# which gives us access to the individual tokens (words, punctuation) and their
# linguistic annotations (part-of-speech tags, dependency labels) predicted
# by the statistical model. See here for the visualization:
# https://explosion.ai/demos/displacy?text=These%20generously%20buttered%20noodles%2C%20sprinkled%20with%20just%20a%20quarter%20cup%20of%20parsley%20for%20color%20and%20freshness%2C%20are%20the%20perfect%20blank%20canvas%20for%20practically%20any%20stew%20or%20braise&model=en_core_web_sm&cpu=0&cph=0
# doc = nlp(text)


# doc = nlp(u"The Empire of Japan aimed to dominate Asia and the " \
#                "Pacific and was already at war with the Republic of China " \
#                "in 1937, but the world war is generally said to have begun on " \
#                "1 September 1939 with the invasion of Poland by Germany and " \
#                "subsequent declarations of war on Germany by France and the United Kingdom. " \
#                "From late 1939 to early 1941, in a series of campaigns and treaties, Germany conquered " \
#                "or controlled much of continental Europe, and formed the Axis alliance with Italy and Japan. " \
#                "Under the Molotov-Ribbentrop Pact of August 1939, Germany and the Soviet Union partitioned and " \
#                "annexed territories of their European neighbours, Poland, Finland, Romania and the Baltic states. " \
#                "The war continued primarily between the European Axis powers and the coalition of the United Kingdom " \
#                "and the British Commonwealth, with campaigns including the North Africa and East Africa campaigns, " \
#                "the aerial Battle of Britain, the Blitz bombing campaign, the Balkan Campaign as well as the " \
#                "long-running Battle of the Atlantic. In June 1941, the European Axis powers launched an invasion " \
#                "of the Soviet Union, opening the largest land theatre of war in history, which trapped the major part " \
#                "of the Axis' military forces into a war of attrition. In December 1941, Japan attacked " \
#                "the United States and European territories in the Pacific Ocean, and quickly conquered much of " \
#                "the Western Pacific.")
#
# pipes_structure = [SequencePipe([FindTokensPipe("VERB/nsubj/*"),
#                                  NamedEntityFilterPipe(),
#                                  NamedEntityExtractorPipe()]),
#                    FindTokensPipe("VERB"),
#                    AnyPipe([SequencePipe([FindTokensPipe("VBD/dobj/NNP"),
#                                           AggregatePipe([NamedEntityFilterPipe("GPE"),
#                                                 NamedEntityFilterPipe("PERSON")]),
#                                           NamedEntityExtractorPipe()]),
#                             SequencePipe([FindTokensPipe("VBD/**/*/pobj/NNP"),
#                                           AggregatePipe([NamedEntityFilterPipe("LOC"),
#                                                 NamedEntityFilterPipe("PERSON")]),
#                                           NamedEntityExtractorPipe()])])]
#
# engine = PipelineEngine(pipes_structure, Context(doc), [0,1,2])
# engine.process()

import neuralcoref

# # nlp = spacy.load('en')
# neuralcoref.add_to_pipe(nlp)
# doc1 = nlp('Angela orders Chicken Biryani. She likes it a lot. It was sp delicious.')
# print(doc1._.coref_clusters)
#
# doc2 = nlp('Angela orders Chicken Biryani. She likes it a lot. It was sp delicious.')
# for ent in doc2.ents:
#     print(ent._.coref_cluster)


# Let's try before using the conversion dictionary:
neuralcoref.add_to_pipe(nlp)
doc = nlp(u'Angela orders Chicken Biryani. She likes it a lot. It was sp delicious.')
print(doc._.coref_clusters)
print(doc._.coref_resolved)

print(doc._.coref_clusters)
print(doc._.coref_clusters[1].mentions)
print(doc._.coref_clusters[1].mentions[-1])
print(doc._.coref_clusters[1].mentions[-1]._.coref_cluster.main)

token = doc[-1]
print(token._.in_coref)
print(token._.coref_clusters)

# span = doc[-1:]
# print(span._.is_coref)
# print(span._.coref_cluster.main)
# print(span._.coref_cluster.main._.coref_cluste)

# >>> [Deepika: [Deepika, She, him, The movie star]]
# >>> 'Deepika has a dog. Deepika loves Deepika. Deepika has always been fond of animals'
# >>> Not very good...

# Here are three ways we can add the conversion dictionary
nlp.remove_pipe("neuralcoref")
neuralcoref.add_to_pipe(nlp, conv_dict={'Chicken Biryani': ['it']})
# or
# nlp.remove_pipe("neuralcoref")
# coref = neuralcoref.NeuralCoref(nlp.vocab, conv_dict={'Deepika': ['woman', 'actress']})
# nlp.add_pipe(coref, name='neuralcoref')
# or after NeuralCoref is already in SpaCy's pipe, by modifying NeuralCoref in the pipeline
# nlp.get_pipe('neuralcoref').set_conv_dict({'Deepika': ['woman', 'actress']})

# Let's try agin with the conversion dictionary:
doc = nlp(u'Angela orders Chicken Biryani. She likes it a lot. It was sp delicious.')
# print(doc._.coref_clusters)
# >>> [Deepika: [Deepika, She, The movie star], a dog: [a dog, him]]
# >>> 'Deepika has a dog. Deepika loves a dog. Deepika has always been fond of animals'
# >>> A lot better!


def get_root(doc):
    # Based on a processed document, we want to find the syntactic root of the
    # sentence. For this example, that should be the verb "are".
    for token in doc:
        if token.dep_ == "ROOT":
            return token


def get_subject(root):
    # If we know the root of the sentence, we can use it to find its subject.
    # Here, we're checking the root's children for a token with the dependency
    # label 'nsubj' (nominal subject).
    for token in root.children:
        if token.dep_ == "nsubj":
            return token


def get_object(root):
    # We also need to look for the object attached to the root. In this case,
    # the dependency parser predicted the object we're looking for ("canvas")
    # as "attr" (attribute), so we're using that. There are various other
    # options, though, so if you want to generalise this script, you'd probably
    # want to check for those as well.
    for token in root.children:
        if token.dep_ == "attr":
            return token


def get_pronoun(token):
    # Based on the subject token, we need to decide which pronoun to use.
    # For example, "noodle" would require "it", whereas "noodles" needs "they".
    # You might also just want to skip singular nouns alltogether and focus
    # on the plurals only, which are much simpler to deal with in English.

    # spaCy currently can't do this out-of-the-box, but there are other
    # rule-based or statistical systems that can do this pretty well. For
    # simplicity, we just mock this up here and always use "they".
    return "they"


def get_subtree(token):
    # Here, we are getting the subtree of a token – for example, if we know
    # that "noodles" is the subject, we can resolve it to the full phrase
    # "These generously buttered noodles, sprinkled with just a quarter cup of
    # parsley for color and freshness".

    # spaCy preserves the whitespace following a token in the `text_with_ws`
    # attribute. This means you'll alwas be able to restore the original text.
    # For example: "Hello world!" (good) vs. "Hello world !" (bad).
    subtree = [t.text_with_ws for t in token.subtree]
    subtree = "".join(subtree)

    # Since our template will place the subject and object in the middle of a
    # sentence, we also want to make sure that the first token starts with a
    # lowercase letter – otherwise we'll end up with things like "love These".
    subtree = subtree[0].lower() + subtree[1:]
    return subtree

# doc = nlp("Angela orders Chicken Biryani. She likes it a lot. It was sp delicious.")
# for token in doc:
#     print(token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])
#     # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#     #         token.shape_, token.is_alpha, token.is_stop)
#
#
# # Let's put this all together!
# root = get_root(doc)
# print("Root:", root)

# subject = get_subject(root)
# print("Subject:", subject)
#
# subject_pronoun = get_pronoun(subject)
# print("Subject pronoun:", subject_pronoun)
#
# obj = get_object(root)
# print("Object:", obj)
#
# subject_subtree = get_subtree(subject)
# print("Subject subtree:", subject_subtree)
#
# object_subtree = get_subtree(obj)
# print("Object subtree:", object_subtree)
#
# print("Result:")
# print(
#     template.format(
#         subject=subject_subtree,
#         pronoun=subject_pronoun,
#         verb=root.text,
#         obj=object_subtree,
#     )
# )