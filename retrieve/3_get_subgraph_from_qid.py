import requests
import json
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
 
def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    try:
        return requests.get(url, params=params)
    except:
        return 'There was and error'
    

def get_name_from_qid(sparql, qid):
    sparql.setQuery(
    """
    SELECT DISTINCT * WHERE {
    wd:"""+qid+""" rdfs:label ?label . 
    FILTER (langMatches( lang(?label), "EN" ) )  
    }
    """
    )
    try:
        ret = sparql.queryAndConvert()
        name = ret["results"]["bindings"][0]['label']['value']
        return name
    except Exception as e:
        print(e)
        return 0
    

def get_dates(sparql, name_id, feature_id, feature_name):
    sparql.setQuery(
    """
    SELECT ?"""+feature_name+""" WHERE {
    wd:"""+name_id+""" p:"""+feature_id+""" ?person.
    OPTIONAL { ?person ps:"""+feature_id+""" ?"""+feature_name+""". }
    }
    """
    )
    feature = ""
    try:
        ret = sparql.queryAndConvert()
        for r in ret["results"]["bindings"]:
            feature = r[feature_name]['value'].split('/')[-1]
    except Exception as e:
        print(e)

    return feature
    

def get_properties(qid):
    # Create parameters
    params = {
                'action': 'wbgetentities',
                'ids':qid, 
                'format': 'json',
                'languages': 'en'
            }
    
    # fetch the API
    data = fetch_wikidata(params)
    data = data.json()
    claims = data["entities"][qid]["claims"]

    properties = {}
    for c in claims:
        try:
            properties[c] = claims[c][0]["mainsnak"]["datavalue"]["value"]["id"]
        except:
            pass

    return properties


if __name__ == "__main__":
    sparql = SPARQLWrapper(
        "https://query.wikidata.org/sparql"
    )
    sparql.setReturnFormat(JSON)
    data = json.load(open('entity_qid_instance.json'))
    property_id = json.load(open('../data/property_id.json'))
    qid_name_map = {}
    output = []
    exceptions = []
 
    names = list(data.keys())
    
    #1. filter out human entities
    human_qid_instance = []
    for name in names:
        qid_instance_list = data[name]
        for qid_instance in qid_instance_list:
            qid = qid_instance[0]
            instance = qid_instance[1]
            if instance == "Q5":
                human_qid_instance.append([name, qid])

    #2. get property of human entity
    for d in tqdm(human_qid_instance):
        name = d[0]
        qid = d[1]
        try:
            property = get_properties(qid)

            # translate property id to property name
            for key in list(property):
                _qid = property[key]
                if _qid[0] == "Q":
                    if _qid in qid_name_map:
                        property[key] = qid_name_map[_qid]
                    else:
                        _name = get_name_from_qid(sparql,_qid)
                        if _name:
                            qid_name_map[_qid] = _name
                            property[key] = _name
                if key in list(property_id):
                    property[property_id[key]] = property.pop(key)
                        
            # input name
            property["name"] = d[0]
            # get dates
            birthdate = get_dates(sparql, qid, "P569", "date_of_birth").split("T")[0]
            deathdate = get_dates(sparql, qid, "P570", "date_of_death").split("T")[0]
            property["date_of_birth"] = birthdate
            property["date_of_death"] = deathdate

            output.append([name, qid, property])
        except:
            exceptions.append([name,qid])

    with open('name_qid_subgraph.json', 'w') as f:
        json.dump(output, f)

    with open('uncollected_name_qid.json', 'w') as f:
        json.dump(exceptions, f)
