import csv
from bisect import bisect_left

import orjson


def clean_single_refs(rerefs):
    for key in rerefs.keys():
        if len(rerefs[key]) < 2:
            del rerefs[key]


def load_rerefs(filename):
    rerefs = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            key = (row["project_name"], row["element"])
            rerefs[key] = rerefs.get(key, []) + [row]
    clean_single_refs(rerefs)
    return rerefs


def load_commits_by_devs():
    commits_by_dev = {}
    with open("commits_by_desenv_liferay.csv") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            key = (row["project_name"], row["author_email"])
            commits_by_dev[key] = commits_by_dev.get(key, []) + [int(row["order"])]
    return commits_by_dev


def index(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


def worked_between(start_order, end_order, ref):
    commits = commits_by_devs[(ref["project_name"], ref["author_email"])]
    start_index = index(commits, int(start_order))
    end_index = index(commits, int(end_order))
    # TODO: Melhorar Criterio
    # Se abs(end_index - start_index) > 1, devemos checar se o dev trabalhou em
    # outra classe ou nao. Se ele trabalhou na mesma durante o intervalo, entao consideramos
    # parte do mesmo batch e nao acabar com o batch
    return abs(end_index - start_index) > 1


def find_batches(key):
    refs = rerefs[key]
    batches = []
    current_batch = []
    element = refs[0]["element"]
    dev = refs[0]["author_email"]
    order = refs[0]["order"]
    for ref in refs:
        if not (ref["element"] == element and ref["author_email"] == dev and not worked_between(order, ref["order"], ref)):
            if len(current_batch) > 1:
                batches.append(current_batch)
            current_batch = []
            element = ref["element"]
            dev = ref["author_email"]
            order = ref["order"]
        current_batch.append(ref)
    if len(current_batch) > 1:
        batches.append(current_batch)
    return batches


rerefs = load_rerefs("liferay_refs.csv")
commits_by_devs = load_commits_by_devs()

i = 1
multiple = 0
all_batches = []
for key in rerefs:
    batches = find_batches(key)
    all_batches += batches
    for batch in batches:
        print("Batch", i)
        i += 1
        orders = set()
        for ref in batch:
            print(ref["element"], ref["author_email"], ref["order"])
            orders.add(ref["order"])
        if len(orders) > 1:
            multiple += 1
        print("")


with open("results/liferay_element_batches_to_import.json", "wb") as output:
    output.write(orjson.dumps(all_batches, option=orjson.OPT_INDENT_2))

# print(len(all_batches))
#
# print(orjson.dumps(all_batches, option=orjson.OPT_INDENT_2))
# print("Batches: %s (%s multiples)" % (len(all_batches), multiple))
