import json


if __name__ == "__main__":
    question_entity_qid = json.load(open("../retrieve/question_entity_qid.json"))
    name_qid_subgraph = json.load(open("../retrieve/name_qid_subgraph3.json"))

    map = {}
    avail_qids = []
    for item in name_qid_subgraph:
        qid = item[1]
        subgraph = item[2]
        map[qid] = subgraph
        avail_qids.append(qid)

    questions_graphs = []
    c = 0
    cc = 0
    for item in question_entity_qid:
        question = item[0][0]
        qid_entities = item[1]

        all_graphs = []
        for qid_entity in qid_entities:
            name = qid_entity[0]
            graphs = [] # all qids that have the same name
            if len(qid_entity[1]) > 0:
                for a in qid_entity[1]:
                    qid = a[0]
                    if qid in avail_qids:
                        graphs.append(map[qid])
            if len(graphs) > 0:
                all_graphs.append(graphs)
            if len(all_graphs) > 1:
                cc = cc + 1
        if len(all_graphs) == 0:
            c = c + 1
        questions_graphs.append([question, all_graphs])

    with open('questions_graph.json', 'w') as f:
        json.dump(questions_graphs, f)

    output = []
    all_display_overlaps = []
    for item in questions_graphs: # at this level, we look at all the each question, and all the graphs provided 
        question = item[0]
        all_graphs = item[1]
        display_overlaps = []
        all_selected_graphs = []

        for graphs in all_graphs: # at this level, we look at the graphs that belong to same name
            selected_graph = []
            highest_overlap = -1

            for graph in graphs: # at this level, we look at each specific graph
                overlap = 0
                value = list(graph.values())

                for v in value: # at this level, we look at each value in the graph
                    if v in question:
                        overlap = overlap + 1
                if overlap > highest_overlap:
                    selected_graph = graph
                    highest_overlap = overlap
            if selected_graph["name"].count(" ") > 0:
                display_overlaps.append(highest_overlap)
                all_selected_graphs.append(selected_graph)
        all_display_overlaps.append(display_overlaps)
        output.append([question, all_selected_graphs])

    with open('questions_selected_graphs.json', 'w') as f:
        json.dump(output, f)

    with open('overlap_stats.json', 'w') as f:
        json.dump(all_display_overlaps, f)





