kb_entities = {
    "Category": ["Chinese", "Indian", "Korean", "Italian", "Mexican"],
    "Attr": ["Good", "Sweet", "Spicy", "Friendly"],
    "Aspect": ["Service", "Take out", "Staff", "Parking"],
    "City": ["Toronto", "Seoul", "New York"],
    "Menu": ["Noodles", "Chicken Biryani", "Kimchi"],
    "Restaurant": ["Silver Spoon", "South Spice", "Little Seoul"]
}
# "", "find", "search", "recommend", "suggest" __ "restaurant|restaurants"
# "", "what", "which" __ "menu|menus"
# We ignore outgoing edge of City as we are only try to resolve restaurant related query, so only incoming
# relationship is applicable
kb_rel = {
    "Category": {"in": ["HAS_CATEGORY"], "out": []},
    "Attr": {"in": ["IS"], "out": ["MENU_ATTR_FOR", "ASPECT_ATTR_FOR"]},
    "Aspect": {"in": ["HAS_ASPECT"], "out": ["IS"]},
    "City": {"in": ["LOCATED_IN"], "out": ["LOCATED_IN"]},
    "Menu": {"in": ["HAS_MENU"], "out": ["IS"]}
}


# text = "Recommend best Chinese restaurant in Toronto or New York and Seoul which serves sweet " \
#        "Noodles and spicy Chicken Biryani with friendly staff and service"
# [
# {'triple': [('?', 'HAS_CATEGORY', 'Chinese')], 'concept': 'Category', 'op': None},
# {'triple': [('?', 'HAS_ASPECT', 'Service'), ('Service', 'IS', '?')], 'concept': 'Aspect', 'op': 'AND'},
# {'triple': [('?', 'HAS_ASPECT', 'Staff'), ('Staff', 'IS', '?'), ('?', 'IS', 'Friendly'), ('Friendly', 'MENU_ATTR_FOR', '?'), ('Friendly', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Aspect', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto'), ('Toronto', 'LOCATED_IN', '?')], 'concept': 'City', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Seoul'), ('Seoul', 'LOCATED_IN', '?')], 'concept': 'City', 'op': 'AND'},
# {'triple': [('?', 'LOCATED_IN', 'New York'), ('New York', 'LOCATED_IN', '?')], 'concept': 'City', 'op': 'OR'},
# {'triple': [('?', 'HAS_MENU', 'Noodles'), ('Noodles', 'IS', '?'), ('?', 'IS', 'Sweet'), ('Sweet', 'MENU_ATTR_FOR', '?'), ('Sweet', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Menu', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Chicken Biryani'), ('Chicken Biryani', 'IS', '?'), ('?', 'IS', 'Spicy'), ('Spicy', 'MENU_ATTR_FOR', '?'), ('Spicy', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Menu', 'op': 'AND'}
# ]
# [
# {'triple': [('?', 'HAS_CATEGORY', 'Chinese')], 'concept': 'Category', 'op': None},
# {'triple': [('?', 'HAS_ASPECT', 'Service')], 'concept': 'Aspect', 'op': 'AND'},
# {'triple': [('?', 'HAS_ASPECT', 'Staff'), ('Friendly', 'ASPECT_ATTR_FOR', '?'), ('Staff', 'IS', 'Friendly')], 'concept': 'Aspect', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto')], 'concept': 'City', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Seoul')], 'concept': 'City', 'op': 'AND'},
# {'triple': [('?', 'LOCATED_IN', 'New York')], 'concept': 'City', 'op': 'OR'},
# {'triple': [('?', 'HAS_MENU', 'Noodles'), ('Sweet', 'MENU_ATTR_FOR', '?'), ('Noodles', 'IS', 'Sweet')], 'concept': 'Menu', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Chicken Biryani'), ('Spicy', 'MENU_ATTR_FOR', '?'), ('Chicken Biryani', 'IS', 'Spicy')], 'concept': 'Menu', 'op': 'AND'}
# ]


# text = "Recommend best Chinese restaurant in Toronto or New York which serves sweet " \
#        "Noodles and spicy Chicken Biryani"
# [
# {'triple': [('?', 'HAS_CATEGORY', 'Chinese')], 'concept': 'Category', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto'), ('Toronto', 'LOCATED_IN', '?')], 'concept': 'City', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'New York'), ('New York', 'LOCATED_IN', '?')], 'concept': 'City', 'op': 'OR'},
# {'triple': [('?', 'HAS_MENU', 'Noodles'), ('Noodles', 'IS', '?'), ('?', 'IS', 'Sweet'), ('Sweet', 'MENU_ATTR_FOR', '?'), ('Sweet', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Menu', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Chicken Biryani'), ('Chicken Biryani', 'IS', '?'), ('?', 'IS', 'Spicy'), ('Spicy', 'MENU_ATTR_FOR', '?'), ('Spicy', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Menu', 'op': 'AND'}
# ]
# [
# {'triple': [('?', 'HAS_CATEGORY', 'Chinese')], 'concept': 'Category', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto')], 'concept': 'City', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'New York')], 'concept': 'City', 'op': 'OR'},
# {'triple': [('?', 'HAS_MENU', 'Noodles'), ('Sweet', 'MENU_ATTR_FOR', '?'), ('Noodles', 'IS', 'Sweet')], 'concept': 'Menu', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Chicken Biryani'), ('Spicy', 'MENU_ATTR_FOR', '?'), ('Chicken Biryani', 'IS', 'Spicy')], 'concept': 'Menu', 'op': 'AND'}
# ]


# text = "Recommend best Chinese restaurant in Toronto or New York which serves sweet" \
#        "Noodles and Chicken Biryani"
# [
# {'triple': [('?', 'HAS_CATEGORY', 'Chinese')], 'concept': 'Category', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto'), ('Toronto', 'LOCATED_IN', '?')], 'concept': 'City', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'New York'), ('New York', 'LOCATED_IN', '?')], 'concept': 'City', 'op': 'OR'},
# {'triple': [('?', 'HAS_MENU', 'Noodles'), ('Noodles', 'IS', '?'), ('?', 'IS', 'Sweet'), ('Sweet', 'MENU_ATTR_FOR', '?'), ('Sweet', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Menu', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Chicken Biryani'), ('Chicken Biryani', 'IS', '?')], 'concept': 'Menu', 'op': 'AND'}
# ]
# [
# {'triple': [('?', 'HAS_CATEGORY', 'Chinese')], 'concept': 'Category', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto')], 'concept': 'City', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'New York')], 'concept': 'City', 'op': 'OR'},
# {'triple': [('?', 'HAS_MENU', 'Noodles'), ('Sweet', 'MENU_ATTR_FOR', '?'), ('Noodles', 'IS', 'Sweet')], 'concept': 'Menu', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Chicken Biryani')], 'concept': 'Menu', 'op': 'AND'}
# ]


# text = "Recommend best Chinese restaurant in Toronto which serves sweet " \
#        "Noodles and spicy Chicken Biryani"
# [
# {'triple': [('?', 'HAS_CATEGORY', 'Chinese')], 'concept': 'Category', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto'), ('Toronto', 'LOCATED_IN', '?')], 'concept': 'City', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Noodles'), ('Noodles', 'IS', '?'), ('?', 'IS', 'Sweet'), ('Sweet', 'MENU_ATTR_FOR', '?'), ('Sweet', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Menu', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Chicken Biryani'), ('Chicken Biryani', 'IS', '?'), ('?', 'IS', 'Spicy'), ('Spicy', 'MENU_ATTR_FOR', '?'), ('Spicy', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Menu', 'op': 'AND'}
# ]
# [
# {'triple': [('?', 'HAS_CATEGORY', 'Chinese')], 'concept': 'Category', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto')], 'concept': 'City', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Noodles'), ('Sweet', 'MENU_ATTR_FOR', '?'), ('Noodles', 'IS', 'Sweet')], 'concept': 'Menu', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Chicken Biryani'), ('Spicy', 'MENU_ATTR_FOR', '?'), ('Chicken Biryani', 'IS', 'Spicy')], 'concept': 'Menu', 'op': 'AND'}
# ]


# text = "Recommend best Chinese restaurant in Toronto which serves sweet Noodles"
# [
# {'triple': [('?', 'HAS_CATEGORY', 'Chinese')], 'concept': 'Category', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto'), ('Toronto', 'LOCATED_IN', '?')], 'concept': 'City', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Noodles'), ('Noodles', 'IS', '?'), ('?', 'IS', 'Sweet'), ('Sweet', 'MENU_ATTR_FOR', '?'), ('Sweet', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Menu', 'op': None}
# ]
# [
# {'triple': [('?', 'HAS_CATEGORY', 'Chinese')], 'concept': 'Category', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto')], 'concept': 'City', 'op': None},
# {'triple': [('?', 'HAS_MENU', 'Noodles'), ('Sweet', 'MENU_ATTR_FOR', '?'), ('Noodles', 'IS', 'Sweet')], 'concept': 'Menu', 'op': None}
# ]


# text = "restaurants in toronto with parking"
# [
# {'triple': [('?', 'HAS_ASPECT', 'Parking'), ('Parking', 'IS', '?')], 'concept': 'Aspect', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto'), ('Toronto', 'LOCATED_IN', '?')], 'concept': 'City', 'op': None}
# ]
# [
# {'triple': [('?', 'HAS_ASPECT', 'Parking')], 'concept': 'Aspect', 'op': None},
# {'triple': [('?', 'LOCATED_IN', 'Toronto')], 'concept': 'City', 'op': None}
# ]


# text = "restaurants with friendly staff"
# [
# {'triple': [('?', 'HAS_ASPECT', 'Staff'), ('Staff', 'IS', '?'), ('?', 'IS', 'Friendly'), ('Friendly', 'MENU_ATTR_FOR', '?'), ('Friendly', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Aspect', 'op': None}
# ]
# [
# {'triple': [('?', 'HAS_ASPECT', 'Staff'), ('Friendly', 'ASPECT_ATTR_FOR', '?'), ('Staff', 'IS', 'Friendly')], 'concept': 'Aspect', 'op': None}
# ]


# text = "restaurants which offers spicy chicken biryani"
# [
# {'triple': [('?', 'HAS_MENU', 'Chicken Biryani'), ('Chicken Biryani', 'IS', '?'), ('?', 'IS', 'Spicy'), ('Spicy', 'MENU_ATTR_FOR', '?'), ('Spicy', 'ASPECT_ATTR_FOR', '?')], 'concept': 'Menu', 'op': None}
# ]
# [
# {'triple': [('?', 'HAS_MENU', 'Chicken Biryani'), ('Spicy', 'MENU_ATTR_FOR', '?'), ('Chicken Biryani', 'IS', 'Spicy')], 'concept': 'Menu', 'op': None}
# ]


# text = "what menus Silver Spoon restaurant offers?"



def kb_concept_lookup(dictionary, string):
    concepts = []

    for key, val in dictionary.items():
        if key != "Attr":
            for concept in val:
                if concept.lower() in string.lower():
                    concept_dict = dict()
                    if key == "Menu":
                        # print(string.index(concept))
                        menu_attr = text[:text.lower().find(concept.lower())].split()[-1]
                        # print(dictionary.get("Attr"))

                        if menu_attr.title() in dictionary.get("Attr"):
                            # print(menu_attr.title())
                            concept_dict[key] = concept.title()
                            concept_dict["Attr"] = menu_attr.title()
                            concept = menu_attr + " " + concept
                        else:
                            concept_dict[key] = concept.title()

                    elif key == "Aspect":
                        aspect_attr = text[:text.lower().find(concept.lower())].split()[-1]

                        if aspect_attr.title() in dictionary.get("Attr"):
                            concept_dict[key] = concept.title()
                            concept_dict["Attr"] = aspect_attr.title()
                            concept = aspect_attr + " " + concept
                        else:
                            concept_dict[key] = concept.title()

                    else:
                        concept_dict[key] = concept.title()

                    concept_op = text[:text.lower().find(concept.lower())].split()[-1]

                    if concept_op.lower() in ["or", "and"]:
                        concept_dict["op"] = concept_op.upper()
                    else:
                        concept_dict["op"] = None

                    concepts.append(concept_dict)
    return concepts


def kb_relation_lookup(dictionary, query_concepts):
    print(dictionary)
    print(query_concepts)

    triples = []
    for query_concept in query_concepts:

        triple_obj = dict()
        triple_obj["triple"] = []
        for key, val in query_concept.items():

            if key == "op":
                triple_obj[key] = val
                triples.append(triple_obj)
                continue

            if key != "Attr":
                triple_obj["concept"] = key

            print(key, val)
            relation = dictionary[key]
            print(relation)

            in_rel = relation["in"]
            if len(in_rel) > 0:
                print(in_rel)
                for rel in in_rel:
                    triple_obj["triple"].append(("?", rel, val))

            out_rel = relation["out"]
            if len(out_rel) > 0:
                print(out_rel)
                for rel in out_rel:
                    triple_obj["triple"].append((val, rel, "?"))

    print(triples)
    # Filter triples
    for triple_obj in triples:

        # Filter ASPECT concept
        if triple_obj["concept"] == "Aspect":

            for triple in list(triple_obj["triple"]):
                if triple[1] == "MENU_ATTR_FOR":
                    triple_obj["triple"].remove(triple)

            has_aspect_attr = False
            for triple in triple_obj["triple"]:
                if triple[1] == "ASPECT_ATTR_FOR":
                    has_aspect_attr = True
                    break

            if not has_aspect_attr:
                for triple in list(triple_obj["triple"]):
                    if triple[1] == "IS":
                        triple_obj["triple"].remove(triple)
            else:
                (sub, obj) = (None, None)
                for triple in list(triple_obj["triple"]):

                    if triple[1] == "IS":

                        if not (triple[0] == "?" or triple[2] == "?"):
                            break

                        if triple[0] == "?":
                            obj = triple[2]
                        else:
                            sub = triple[0]

                        triple_obj["triple"].remove(triple)

                    if sub is not None and obj is not None:
                        triple_obj["triple"].append((sub, "IS", obj))
                        (sub, obj) = (None, None)

        # Filter Menu Concept
        if triple_obj["concept"] == "Menu":

            for triple in list(triple_obj["triple"]):
                if triple[1] == "ASPECT_ATTR_FOR":
                    triple_obj["triple"].remove(triple)

            has_menu_attr = False
            for triple in triple_obj["triple"]:
                if triple[1] == "MENU_ATTR_FOR":
                    has_menu_attr = True
                    break

            if not has_menu_attr:
                for triple in list(triple_obj["triple"]):
                    if triple[1] == "IS":
                        triple_obj["triple"].remove(triple)
            else:
                (sub, obj) = (None, None)
                for triple in list(triple_obj["triple"]):

                    if triple[1] == "IS":

                        if not (triple[0] == "?" or triple[2] == "?"):
                            break

                        if triple[0] == "?":
                            obj = triple[2]
                        else:
                            sub = triple[0]

                        triple_obj["triple"].remove(triple)

                    if sub is not None and obj is not None:
                        triple_obj["triple"].append((sub, "IS", obj))
                        (sub, obj) = (None, None)

        # Filter City Concept
        if triple_obj["concept"] == "City":
            for triple in list(triple_obj["triple"]):
                if not triple[0] == "?":
                    triple_obj["triple"].remove(triple)

    print(triples)

query_concepts = kb_concept_lookup(kb_entities, text)
print(query_concepts)

kb_relation_lookup(kb_rel, query_concepts)