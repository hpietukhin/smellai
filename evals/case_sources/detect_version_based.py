import orjson
from py2neo import Graph


def clean_single_refs(rerefs):
    for key in rerefs.keys():
        if len(rerefs[key]) < 2:
            del rerefs[key]


def find_batches():
    batches = []
    cypher = """
        match (r:Refactoring)-->(c:Commit) 
        return c as commit, collect(r) as refs
        order by c.order
    """
    graph = Graph(password="boil2.eat")
    data = graph.data(cypher)
    for row in data:
        batch = []
        for refactoring in row["refs"]:
            batch.append({
                "commit_id": row["commit"]["hash_id"],
                "type": refactoring["type"],
                "order": row["commit"]["order"],
                "ref_id": refactoring["hash_id"]
            })
        if len(batch) > 1:
            batches.append(batch)
    return batches


all_batches = find_batches()

json_result = orjson.dumps(all_batches, option=orjson.OPT_INDENT_2)
with open("results/version_batches_to_import.json", "wb") as f:
    f.write(json_result)
