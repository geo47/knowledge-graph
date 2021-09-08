from neo4j import GraphDatabase
import pandas as pd

class GraphDB:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def print_greeting(self, message):
        with self.driver.session() as session:
            greeting = session.write_transaction(self._create_and_return_greeting, message)
            print(greeting)

    @staticmethod
    def _create_and_return_greeting(tx, message):
        result = tx.run("CREATE (a:Greeting) "
                        "SET a.message = $message "
                        "RETURN a.message + ', from node ' + id(a)", message=message)
        return result.single()[0]

    def get_embedding_result(self):
        with self.driver.session() as session:
            result = session.write_transaction(self._get_embedding_result)
            # for record in result:
            #     print(record)
            return result

    @staticmethod
    def _get_embedding_result(tx):
        result = tx.run("MATCH (rest:Restaurant)-[:HAS_MENU]->(menu) "
                        "WHERE menu.name IN $menus "
                        "RETURN rest.name AS restaurant, rest.embeddingNode2Vec AS embedding, menu.name AS menu",
                        {"menus": ["Chicken"]})
        # for record in result:
        #     print(record)
        X = pd.DataFrame([dict(record) for record in result])
        return X
        # return result.single()[0]

    ######################################### Creating Nodes ############################################

    # `User` Node
    def create_user(self, user_id, name, gender, age, fans, review_count, average_stars):
        with self.driver.session() as session:
            user = session.write_transaction(self._create_user, user_id, name, gender, age, fans, review_count,
                                             average_stars)
            print(user)

    @staticmethod
    def _create_user(tx, user_id, name, gender, age, fans, review_count, average_stars):
        result = tx.run("CREATE (user:User) "
                        "SET user.user_id = $user_id, user.name = $name, user.gender = $gender, user.age = $age, "
                        "user.fans = $fans, user.review_count = $review_count, user.average_stars = $average_stars "
                        "RETURN user.user_id + ', from node ' + id(user)", user_id=user_id, name=name, gender=gender,
                        age=age, fans=fans, review_count=review_count, average_stars=average_stars)
        return result.single()[0]

    # `Restaurant` Node
    def create_restaurant(self, rest_id, name, address, postal_code, rating, restaurant_photos):
        with self.driver.session() as session:
            restaurant = session.write_transaction(self._create_restaurant, rest_id, name,
                                                   address, postal_code, rating, restaurant_photos)
            print(restaurant)

    @staticmethod
    def _create_restaurant(tx, rest_id, name, address, postal_code, rating, restaurant_photos):
        result = tx.run("CREATE (rest:Restaurant) "
                        "SET rest.rest_id = $rest_id, rest.name = $name, "
                        "rest.address = $address, rest.postal_code = $postal_code, rest.rating = $rating, "
                        "rest.restaurant_photos = $restaurant_photos RETURN rest.rest_id + ', from node ' + id(rest)",
                        rest_id=rest_id, name=name, address=address, postal_code=postal_code, rating=rating,
                        restaurant_photos=restaurant_photos)
        return result.single()[0]

    # `Category` Node
    def create_category(self, category_id, name):
        with self.driver.session() as session:
            aspect = session.write_transaction(self._create_category, category_id, name)
            print(aspect)

    @staticmethod
    def _create_category(tx, category_id, name):
        result = tx.run("MERGE (category:Category {category_id: $category_id, name: $name}) "
                        "RETURN category.category_id + ', from node ' + id(category)", category_id=category_id,
                        name=name)
        return result.single()[0]

    # `Review` Node
    def create_review(self, review_id, text):
        with self.driver.session() as session:
            review = session.write_transaction(self._create_review, review_id, text)
            print(review)

    @staticmethod
    def _create_review(tx, review_id, text):
        result = tx.run("CREATE (rev:Review) "
                        "SET rev.review_id = $review_id, rev.text = $text "
                        "RETURN rev.review_id + ', from node ' + id(rev)", review_id=review_id, text=text)
        return result.single()[0]

    # `Menu` Node
    def create_menu(self, menu_id, name):
        with self.driver.session() as session:
            menu = session.write_transaction(self._create_menu, menu_id, name)
            print(menu)

    @staticmethod
    def _create_menu(tx, menu_id, name):
        result = tx.run("MERGE (menu:Menu {menu_id: $menu_id, name: $name}) "
                        "RETURN menu.menu_id + ', from node ' + id(menu)", menu_id=menu_id, name=name)
        return result.single()[0]

    # `Aspect` Node
    def create_aspect(self, aspect_id, name):
        with self.driver.session() as session:
            aspect = session.write_transaction(self._create_aspect, aspect_id, name)
            print(aspect)

    @staticmethod
    def _create_aspect(tx, aspect_id, name):
        result = tx.run("MERGE (aspect:Aspect {aspect_id: $aspect_id, name: $name}) "
                        "RETURN aspect.aspect_id + ', from node ' + id(aspect)", aspect_id=aspect_id, name=name)
        return result.single()[0]

    # `Attr` Node
    def create_attr(self, attr_id, name):
        with self.driver.session() as session:
            aspect = session.write_transaction(self._create_attr, attr_id, name)
            print(aspect)

    @staticmethod
    def _create_attr(tx, attr_id, name):
        result = tx.run("MERGE (attr:Attr {attr_id: $attr_id, name: $name}) "
                        "RETURN attr.attr_id + ', from node ' + id(attr)", attr_id=attr_id, name=name)
        return result.single()[0]

    # `City` Node
    def create_city(self, city_id, name):
        with self.driver.session() as session:
            city = session.write_transaction(self._create_city, city_id, name)
            print(city)

    @staticmethod
    def _create_city(tx, city_id, name):
        result = tx.run("MERGE (city:City {city_id: $city_id, name: $name}) "
                        "RETURN city.city_id + ', from node ' + id(city)", city_id=city_id, name=name)
        return result.single()[0]

    # `State` Node
    def create_state(self, state_id, name):
        with self.driver.session() as session:
            state = session.write_transaction(self._create_state, state_id, name)
            print(state)

    @staticmethod
    def _create_state(tx, state_id, name):
        result = tx.run("MERGE (state:State {state_id: $state_id, name: $name}) "
                        "RETURN state.state_id + ', from node ' + id(state)", state_id=state_id, name=name)
        return result.single()[0]

    # `Country` Node
    def create_country(self, country_id, name):
        with self.driver.session() as session:
            country = session.write_transaction(self._create_country, country_id, name)
            print(country)

    @staticmethod
    def _create_country(tx, country_id, name):
        result = tx.run("MERGE (country:Country {country_id: $country_id, name: $name}) "
                        "RETURN country.country_id + ', from node ' + id(country)", country_id=country_id,
                        name=name)
        return result.single()[0]

    ######################################### Creating Relations ############################################

    # (User, User): HAS_FRIEND
    def create_user_has_friend_relation(self, user_id, friend_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_user_has_friend_relation, user_id, friend_id)
            print(relation)

    @staticmethod
    def _create_user_has_friend_relation(tx, user_id, friend_id):
        result = tx.run("MATCH (user:User {user_id: $user_id}), "
                        "(friend:User {user_id: $friend_id}) "
                        "MERGE (user)-[r:HAS_FRIEND]->(friend) "
                        "RETURN user.name, type(r), friend.name", user_id=user_id, friend_id=friend_id)
        return result.single()[0]

    # (Restaurant, Review): HAS_REVIEW
    def create_restaurant_has_category_relation(self, rest_id, category_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_restaurant_has_category_relation, rest_id, category_id)
            print(relation)

    @staticmethod
    def _create_restaurant_has_category_relation(tx, rest_id, category_id):
        result = tx.run("MATCH (rest:Restaurant {rest_id: $rest_id}), "
                        "(category:Category {category_id: $category_id}) "
                        "MERGE (rest)-[r:HAS_CATEGORY]->(category) "
                        "RETURN rest.name, type(r), category.category_id", rest_id=rest_id, category_id=category_id)
        return result.single()[0]

    # (Restaurant, Review): HAS_REVIEW
    def create_restaurant_has_review_relation(self, rest_id, review_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_restaurant_has_review_relation, rest_id, review_id)
            print(relation)

    @staticmethod
    def _create_restaurant_has_review_relation(tx, rest_id, review_id):
        result = tx.run("MATCH (rest:Restaurant {rest_id: $rest_id}), "
                        "(rev:Review {review_id: $review_id}) "
                        "MERGE (rest)-[r:HAS_REVIEW]->(rev) "
                        "RETURN rest.name, type(r), rev.id", rest_id=rest_id, review_id=review_id)
        return result.single()[0]

    # (User, Review): WROTE
    def create_user_write_review_relation(self, user_id, review_id, date):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_user_write_review_relation, user_id, review_id, date)
            print(relation)

    @staticmethod
    def _create_user_write_review_relation(tx, user_id, review_id, date):
        result = tx.run("MATCH (user:User {user_id: $user_id}), "
                        "(rev:Review {review_id: $review_id}) "
                        "MERGE (user)-[r:WRITE_REVIEW {date: $date}]->(rev) "
                        "RETURN user.name, type(r), rev.id", user_id=user_id, review_id=review_id, date=date)
        return result.single()[0]

    # (User, Restaurant): VISIT
    def create_user_visit_restaurant_relation(self, user_id, rest_id, count):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_user_visit_restaurant_relation, user_id, rest_id,
                                                 count)
            print(relation)

    @staticmethod
    def _create_user_visit_restaurant_relation(tx, user_id, rest_id, count):
        result = tx.run("MATCH (user:User {user_id: $user_id}), "
                        "(rest:Restaurant {rest_id: $rest_id}) "
                        "MERGE (user)-[r:VISIT {relWeight: $count}]->(rest) "
                        "RETURN user.name, type(r), rest.name", user_id=user_id, rest_id=rest_id, count=count)
        return result.single()[0]

    # (User, Restaurant): RATE
    def create_user_rate_restaurant_relation(self, user_id, rest_id, star):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_user_rate_restaurant_relation, user_id, rest_id,
                                                 star)
            print(relation)

    @staticmethod
    def _create_user_rate_restaurant_relation(tx, user_id, rest_id, star):
        result = tx.run("MATCH (user:User {user_id: $user_id}), "
                        "(rest:Restaurant {rest_id: $rest_id}) "
                        "MERGE (user)-[r:RATE {relWeight: $star}]->(rest) "
                        "RETURN user.name, type(r), rest.name", user_id=user_id, rest_id=rest_id, star=star)
        return result.single()[0]

    # (Restaurant, City): IN_CITY
    def create_restaurant_city_relation(self, rest_id, city_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_restaurant_city_relation, rest_id, city_id)
            print(relation)

    @staticmethod
    def _create_restaurant_city_relation(tx, rest_id, city_id):
        result = tx.run("MATCH (rest:Restaurant {rest_id: $rest_id}), "
                        "(city:City {city_id: $city_id}) "
                        "MERGE (rest)-[r:LOCATED_IN]->(city) "
                        "RETURN rest.name, type(r), city.name", rest_id=rest_id, city_id=city_id)
        return result.single()[0]

    # (City, State): IN_STATE
    def create_city_state_relation(self, city_id, state_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_city_state_relation, city_id, state_id)
            print(relation)

    @staticmethod
    def _create_city_state_relation(tx, city_id, state_id):
        result = tx.run("MATCH (city:City {city_id: $city_id}), "
                        "(state:State {state_id: $state_id}) "
                        "MERGE (city)-[r:LOCATED_IN]->(state) "
                        "RETURN city.name, type(r), state.name", city_id=city_id, state_id=state_id)
        return result.single()[0]

    # (State, Country): IN_COUNTRY
    def create_state_country_relation(self, state_id, country_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_state_country_relation, state_id, country_id)
            print(relation)

    @staticmethod
    def _create_state_country_relation(tx, state_id, country_id):
        result = tx.run("MATCH (state:State {state_id: $state_id}), "
                        "(country:Country {country_id: $country_id}) "
                        "MERGE (state)-[r:LOCATED_IN]->(country) "
                        "RETURN state.name, type(r), country.name", state_id=state_id, country_id=country_id)
        return result.single()[0]

    # (City, Country): IN_COUNTRY
    def create_city_country_relation(self, city_id, country_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_city_country_relation, city_id, country_id)
            print(relation)

    @staticmethod
    def _create_city_country_relation(tx, city_id, country_id):
        result = tx.run("MATCH (city:City {city_id: $city_id}), "
                        "(country:Country {country_id: $country_id}) "
                        "MERGE (city)-[r:LOCATED_IN]->(country) "
                        "RETURN city.name, type(r), country.name", city_id=city_id, country_id=country_id)
        return result.single()[0]

    # (Restaurant, Menu): HAS_MENU
    def create_restaurant_has_menu_relation(self, rest_id, menu_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_restaurant_has_menu_relation, rest_id, menu_id)
            print(relation)

    @staticmethod
    def _create_restaurant_has_menu_relation(tx, rest_id, menu_id):
        result = tx.run("MATCH (rest:Restaurant {rest_id: $rest_id}), "
                        "(menu:Menu {menu_id: $menu_id}) "
                        "MERGE (rest)-[r:HAS_MENU]->(menu) "
                        "RETURN rest.name, type(r), menu.name", rest_id=rest_id, menu_id=menu_id)
        return result.single()[0]

    # (Restaurant, Aspect): HAS_ASPECT
    def create_restaurant_has_aspect_relation(self, rest_id, aspect_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_restaurant_has_aspect_relation, rest_id, aspect_id)
            print(relation)

    @staticmethod
    def _create_restaurant_has_aspect_relation(tx, rest_id, aspect_id):
        result = tx.run("MATCH (rest:Restaurant {rest_id: $rest_id}), "
                        "(aspect:Aspect {aspect_id: $aspect_id}) "
                        "MERGE (rest)-[r:HAS_ASPECT]->(aspect) "
                        "RETURN rest.name, type(r), aspect.name", rest_id=rest_id, aspect_id=aspect_id)
        return result.single()[0]

    # (User, Menu): ORDER
    def create_user_order_menu_relation(self, user_id, menu_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_user_order_menu_relation, user_id, menu_id)
            print(relation)

    @staticmethod
    def _create_user_order_menu_relation(tx, user_id, menu_id):
        result = tx.run("MATCH (user:User {user_id: $user_id}), "
                        "(menu:Menu {menu_id: $menu_id}) "
                        "MERGE (user)-[r:ORDER]->(menu) "
                        "RETURN user.name, type(r), menu.name", user_id=user_id, menu_id=menu_id)
        return result.single()[0]

    # (User, Menu): Open relation
    def create_user_menu_relation(self, user_id, menu_id, rel):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_user_menu_relation, user_id, menu_id, rel)
            print(relation)

    @staticmethod
    def _create_user_menu_relation(tx, user_id, menu_id, rel):
        result = tx.run("MATCH (user:User {user_id: $user_id}), "
                        "(menu:Menu {menu_id: $menu_id}) "
                        "MERGE (user)-[r:"+rel+"]->(menu) "
                        "RETURN user.name, type(r), menu.name", user_id=user_id, menu_id=menu_id)
        return result.single()[0]

    # (Restaurant, Attr): IS
    def create_restaurant_is_attr_relation(self, rest_id, attr_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_restaurant_is_attr_relation, rest_id, attr_id)
            print(relation)

    @staticmethod
    def _create_restaurant_is_attr_relation(tx, rest_id, attr_id):
        result = tx.run("MATCH (rest:Restaurant {rest_id: $rest_id}), "
                        "(attr:Attr {attr_id: $attr_id}) "
                        "MERGE (rest)-[r:IS]->(attr) "
                        "RETURN rest.name, type(r), attr.name", rest_id=rest_id, attr_id=attr_id)
        return result.single()[0]

    # (Restaurant, Attr): Open relation
    def create_restaurant_attr_relation(self, rest_id, attr_id, rel):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_restaurant_attr_relation, rest_id, attr_id, rel)
            print(relation)

    @staticmethod
    def _create_restaurant_attr_relation(tx, rest_id, attr_id, rel):
        result = tx.run("MATCH (rest:Restaurant {rest_id: $rest_id}), "
                        "(attr:Attr {attr_id: $attr_id}) "
                        "MERGE (rest)-[r:"+rel+"]->(attr) "
                        "RETURN rest.name, type(r), attr.name", rest_id=rest_id, attr_id=attr_id)
        return result.single()[0]

    # (Menu, Attr): IS
    def create_menu_is_attr_relation(self, menu_id, attr_id):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_menu_is_attr_relation, menu_id, attr_id)
            print(relation)

    @staticmethod
    def _create_menu_is_attr_relation(tx, menu_id, attr_id):
        result = tx.run("MATCH (menu:Menu {menu_id: $menu_id}), "
                        "(attr:Attr {attr_id: $attr_id}) "
                        "MERGE (menu)-[r:IS]->(attr) "
                        "RETURN menu.name, type(r), attr.name", menu_id=menu_id, attr_id=attr_id)
        return result.single()[0]

    # (Menu, Attr): Open relation
    def create_menu_attr_relation(self, menu_id, attr_id, rel):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_menu_attr_relation, menu_id, attr_id, rel)
            print(relation)

    @staticmethod
    def _create_menu_attr_relation(tx, menu_id, attr_id, rel):
        result = tx.run("MATCH (menu:Menu {menu_id: $menu_id}), "
                        "(attr:Attr {attr_id: $attr_id}) "
                        "MERGE (menu)-[r:"+rel+"]->(attr) "
                        "RETURN menu.name, type(r), attr.name", menu_id=menu_id, attr_id=attr_id, rel=rel)
        return result.single()[0]

    # (Aspect, Attr): IS
    def create_aspect_is_attr_relation(self, aspect_id, attr):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_aspect_is_attr_relation, aspect_id, attr)
            print(relation)

    @staticmethod
    def _create_aspect_is_attr_relation(tx, aspect_id, attr_id):
        result = tx.run("MATCH (aspect:Aspect {aspect_id: $aspect_id}), "
                        "(attr:Attr {attr_id: $attr_id}) "
                        "MERGE (aspect)-[r:IS]->(attr) "
                        "RETURN aspect.name, type(r), attr.name", aspect_id=aspect_id, attr_id=attr_id)
        return result.single()[0]

    # (Aspect, Attr): Open relation
    def create_aspect_attr_relation(self, aspect_id, attr_id, rel):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_aspect_attr_relation, aspect_id, attr_id, rel)
            print(relation)

    @staticmethod
    def _create_aspect_attr_relation(tx, aspect_id, attr_id, rel):
        result = tx.run("MATCH (aspect:Aspect {aspect_id: $aspect_id}), "
                        "(attr:Attr {attr_id: $attr_id}) "
                        "MERGE (aspect)-[r:"+rel+"]->(attr) "
                        "RETURN aspect.name, type(r), attr.name", aspect_id=aspect_id, attr_id=attr_id, rel=rel)
        return result.single()[0]

    # (Attr, Rest): Extra relation between attribute and restaurant to show the attribute property
    # of Menu or Aspect belongs to particular Restaurant
    def create_attr_rest_relation(self, attr_id, rest_id, rel):
        with self.driver.session() as session:
            relation = session.write_transaction(self._create_attr_rest_relation, attr_id, rest_id, rel)
            print(relation)

    @staticmethod
    def _create_attr_rest_relation(tx, attr_id, rest_id, rel):
        result = tx.run("MATCH (attr:Attr {attr_id: $attr_id}), "
                        "(rest:Restaurant {rest_id: $rest_id}) "
                        "MERGE (attr)-[r:" + rel + "]->(rest) "
                        "RETURN attr.name, type(r), rest.name", attr_id=attr_id, rest_id=rest_id,  rel=rel)
        return result.single()[0]