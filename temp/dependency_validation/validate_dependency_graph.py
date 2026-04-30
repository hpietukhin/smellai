from __future__ import annotations
import ast, pathlib, re, json
root=pathlib.Path('.').resolve()
exclude={'.venv','.git','__pycache__','.ruff_cache','.pytest_cache','docs','notebooks','temp'}
py_files=sorted(p for p in pathlib.Path('.').rglob('*.py') if not any(part in exclude for part in p.parts) and 'tests' not in p.parts)
# map module paths to file paths
module_to_file={}
package_dirs=set()
for p in py_files:
    rel=p.with_suffix('').as_posix()
    mod=rel.replace('/','.')
    if p.name=='__init__.py':
        mod='.'.join(p.parent.parts)
        package_dirs.add('.'.join(p.parent.parts))
    module_to_file[mod]=p.as_posix()
# roots are top dirs and single files
local_roots={p.parts[0] for p in py_files}

def resolve_import(src:pathlib.Path, node):
    mods=[]
    if isinstance(node, ast.Import):
        mods=[a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            # relative to package of src
            parts=list(src.with_suffix('').parts)
            if src.name=='__init__.py': parts=list(src.parent.parts)
            else: parts=parts[:-1]
            base=parts[:max(0,len(parts)-node.level+1)]
            if node.module:
                base += node.module.split('.')
            base_mod='.'.join(base)
            # module itself and possible imported submodules
            mods=[base_mod]
            for alias in node.names:
                if alias.name!='*': mods.append(base_mod+'.'+alias.name if base_mod else alias.name)
        else:
            mods=[node.module or '']
            base=node.module or ''
            for alias in node.names:
                if alias.name!='*': mods.append(base+'.'+alias.name if base else alias.name)
    for mod in mods:
        parts=mod.split('.') if mod else []
        if not parts or parts[0] not in local_roots: continue
        # try longest prefix that maps to a module file, then package __init__
        for i in range(len(parts),0,-1):
            cand='.'.join(parts[:i])
            if cand in module_to_file:
                yield module_to_file[cand]; break

edges=set(); parse_errors=[]
for p in py_files:
    try: tree=ast.parse(p.read_text(), filename=str(p))
    except Exception as e:
        parse_errors.append((p.as_posix(),repr(e))); continue
    src=p.as_posix()
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            for tgt in resolve_import(p,n):
                if tgt!=src: edges.add((src,tgt))
html=pathlib.Path('docs/dependency_graph.html').read_text()
html_nodes=set(re.findall(r'id:\s*"([^"]+\.py)"', html))
html_edges=set(re.findall(r'\{\s*source:\s*"([^"]+\.py)"\s*,\s*target:\s*"([^"]+\.py)"', html))
actual_nodes={p.as_posix() for p in py_files if p.name!='__init__.py'}
actual_edges={(a,b) for a,b in edges if not a.endswith('/__init__.py') and not b.endswith('/__init__.py')}
# focused diffs for nodes present in doc (avoid penalizing intentionally omitted scripts)
missing_edges=sorted(actual_edges - html_edges)
stale_edges=sorted(html_edges - actual_edges)
missing_nodes=sorted(actual_nodes - html_nodes)
stale_nodes=sorted(html_nodes - actual_nodes)
summary={
 'actual_nodes':len(actual_nodes), 'html_nodes':len(html_nodes), 'missing_nodes':len(missing_nodes), 'stale_nodes':len(stale_nodes),
 'actual_edges':len(actual_edges), 'html_edges':len(html_edges), 'missing_edges':len(missing_edges), 'stale_edges':len(stale_edges),
 'parse_errors':parse_errors,
}
print(json.dumps(summary, indent=2))
print('\nMISSING_NODES')
print('\n'.join(missing_nodes[:200]))
print('\nSTALE_NODES')
print('\n'.join(stale_nodes))
print('\nMISSING_EDGES')
print('\n'.join(f'{a} -> {b}' for a,b in missing_edges[:300]))
print('\nSTALE_EDGES')
print('\n'.join(f'{a} -> {b}' for a,b in stale_edges[:300]))
