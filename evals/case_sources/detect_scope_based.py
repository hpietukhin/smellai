import csv
from bisect import bisect_left

import orjson


def load_commits_by_devs():
    commits_by_dev = {}
    with open("rerefs/commits_and_devs.csv") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            key = (row["project_name"], row["author_email"])
            commits_by_dev[key] = commits_by_dev.get(key, []) + [int(row["order"])]
    return commits_by_dev


def index(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    #print a, x, i
    raise ValueError


def worked_between(start_order, end_order, project, developer):
    commits = commits_by_devs[(project.lower(), developer)]
    start_index = index(commits, int(start_order))
    end_index = index(commits, int(end_order))
    
    return abs(end_index - start_index) > 1


def load_refactorings(file_name):
    with open(file_name, "rb") as refs_file:
        return orjson.loads(refs_file.read())


class Batch:
    GENERAL_ID = 20000

    def __init__(self, developer):
        self.id = Batch.GENERAL_ID
        Batch.GENERAL_ID += 1

        self.developer = developer
        self.refactorings = []
        self.elements = set()
        self.last_commit_order = 0

    def merge(self, batch):
        self.last_commit_order = max(self.last_commit_order, batch.last_commit_order)
        for refactoring in batch.refactorings:
            self.add_refactoring(refactoring, batch.elements, batch.last_commit_order)

    def is_in(self, refactoring):
        for ref in self.refactorings:
            if ref["hash_id"] == refactoring["hash_id"]:
                return True
        return False

    def add_refactoring(self, refactoring, elements, commit_order):
        if self.is_in(refactoring):
            return
        self.last_commit_order = max(self.last_commit_order, commit_order)
        self.refactorings.append(refactoring)
        for el in elements:
            self.elements.add(el)


def find_candidates(batches, row):
    project = row["project"]["name"]
    commit = row["commit"]
    developer = commit["author_email"]
    candidates = []
    elements = set(row["elements"])
    for batch in batches:
        if batch.developer != developer:
            continue
        if not worked_between(batch.last_commit_order, commit["order"], project, developer):
            for el in elements:
                if el in batch.elements:
                    candidates.append(batch)
                    break
    return candidates


def print_batches(batches):
    for k, b in enumerate(batches):
        print("Batch", k)
        print("elements:", list(b.elements))
        for ref in b.refactorings:
            print(ref)
        print("")


def merge_candidates(batches_by_project, project, candidates):
    stored_batches = batches_by_project[project]
    will_remain = candidates[0]
    for batch in candidates:
        if batch is will_remain:
            continue
        will_remain.merge(batch)
        del stored_batches[batch.id]
    return will_remain


def detect_batches(raw_data):
    last_project = "AndroidBootstrap_android-bootstrap"
    merges = 0
    batches_by_project = {}
    for row in raw_data:
        refactoring = row["refactoring"]
        refactoring["commit"] = row["commit"]
        project = row["project"]["name"]
        if project != last_project:
            print(project)
            last_project = project
        batches = batches_by_project.get(project, {})
        candidates = find_candidates(batches.values(), row)
        if len(candidates) > 1:
            merges += 1
            merged = merge_candidates(batches_by_project, project, candidates)
            merged.add_refactoring(refactoring, row["elements"], row["commit"]["order"])
        elif len(candidates) == 1:
            batch = candidates[0]
            batch.add_refactoring(refactoring, row["elements"], row["commit"]["order"])
        else:
            batch = Batch(row["commit"]["author_email"])
            batch.add_refactoring(refactoring, row["elements"], row["commit"]["order"])
            batches[batch.id] = batch
        batches_by_project[project] = batches

    total_batches = 0
    for project in batches_by_project:
        batches = batches_by_project[project].values()
        batches_by_project[project] = filter(lambda b: len(b.refactorings) > 1, batches)
        total_batches += len(batches_by_project[project])
    print(merges, total_batches)
    return batches_by_project


def convert_to_import_format(batches_by_project):
    all_batches = []
    for project in batches_by_project:
        for batch_obj in batches_by_project[project]:
            batch_arr = []
            for refactoring in batch_obj.refactorings:
                batch_arr.append({
                    "commit_id": refactoring["commit"]["hash_id"],
                    "ref_id": refactoring["hash_id"],
                    "type": refactoring["type"],
                    "order": refactoring["commit"]["order"],
                })
            all_batches.append(batch_arr)
    return all_batches


commits_by_devs = load_commits_by_devs()
refactorings = load_refactorings("input/refactorings_and_all_elements.json")
batches_by_project = detect_batches(refactorings)

with open("results/scope_batches_to_import.json", "wb") as output:
    output.write(orjson.dumps(convert_to_import_format(batches_by_project), option=orjson.OPT_INDENT_2))
