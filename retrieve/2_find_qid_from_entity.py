from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import json

def get_qid(sparql, entity_list):
    entity_qid_instance_map = {}
    for entity in tqdm(entity_list):
        try:
            qid_instance = []
            sparql.setQuery("""
                SELECT DISTINCT ?qid ?instance_of WHERE {
                ?item rdfs:label '"""+entity+"""'@en.
                BIND(STRAFTER(STR(?item), STR(<http://www.wikidata.org/entity/>)) AS ?qid)
                OPTIONAL { ?item wdt:P31 ?instance_of. }
                }
                """
            )

            ret = sparql.queryAndConvert()
            for r in ret["results"]["bindings"]:
                try:
                    qid = r['qid']['value']
                    instance = r['instance_of']['value'].rsplit("/",1)[1]
                except Exception as e:
                    pass
                qid_instance.append((qid, instance))
            entity_qid_instance_map[entity] = qid_instance
        except:
            print(entity)

    return entity_qid_instance_map

if __name__ == "__main__":
    sparql = SPARQLWrapper(
        "https://query.wikidata.org/sparql"
    )
    sparql.setReturnFormat(JSON)
    data = json.load(open('retrieved_entities.json'))
    entity_list = []
    for d in data:
        entities = d[1]
        for entity in entities:
            if entity not in entity_list:
                entity_list.append(entity)
    entity_qid_instance_map = get_qid(sparql, entity_list)

    with open('entity_qid_instance.json', 'w') as f:
        json.dump(entity_qid_instance_map, f)

    output = []
    for d in data:
        question = d[0]
        entities = d[1]
        entity_qid_instance = []
        for entity in entities:
            if entity in entity_qid_instance_map:
                entity_qid_instance.append([entity, entity_qid_instance_map[entity]])

        output.append([question, entity_qid_instance])

    with open('question_entity_qid.json', 'w') as f:
        json.dump(output, f)   