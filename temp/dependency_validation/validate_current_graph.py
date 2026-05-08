from __future__ import annotations
import ast
import pathlib
import re
import json
# Same static import model used to renew docs/dependency_graph.html.
exclude={'.venv','.git','__pycache__','.ruff_cache','.pytest_cache','docs','notebooks','temp'}
py_files=sorted(p for p in pathlib.Path('.').rglob('*.py') if not any(part in exclude for part in p.parts) and 'tests' not in p.parts)
module_to_file={}
for p in py_files:
    mod='.'.join(p.parent.parts) if p.name=='__init__.py' else p.with_suffix('').as_posix().replace('/','.')
    module_to_file[mod]=p.as_posix()
local_roots={p.parts[0] for p in py_files}
reexports={}
for p in py_files:
    if p.name!='__init__.py': continue
    pkg='.'.join(p.parent.parts)
    try: tree=ast.parse(p.read_text())
    except Exception: continue
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module:
            if n.level:
                parts=list(p.parent.parts)
                base='.'.join(parts[:max(0,len(parts)-n.level+1)] + n.module.split('.'))
            else: base=n.module
            if base in module_to_file:
                for a in n.names: reexports[(pkg,a.asname or a.name)]=module_to_file[base]

def resolve(src,n):
    mods=[]; aliases=[]
    if isinstance(n, ast.Import): mods=[a.name for a in n.names]
    elif isinstance(n, ast.ImportFrom):
        if n.level:
            parts=list(src.parent.parts if src.name=='__init__.py' else src.with_suffix('').parts[:-1])
            base_parts=parts[:max(0,len(parts)-n.level+1)]
            if n.module: base_parts += n.module.split('.')
            base='.'.join(base_parts)
        else: base=n.module or ''
        mods=[base] if base else []
        aliases=[a.asname or a.name for a in n.names if a.name!='*']
        mods += [(base+'.'+a if base else a) for a in aliases]
        for a in aliases:
            if (base,a) in reexports: yield reexports[(base,a)]
    for mod in mods:
        parts=mod.split('.') if mod else []
        if not parts or parts[0] not in local_roots: continue
        for i in range(len(parts),0,-1):
            cand='.'.join(parts[:i])
            if cand in module_to_file: yield module_to_file[cand]; break
edges=set(); parse_errors=[]
for p in py_files:
    try: tree=ast.parse(p.read_text(), filename=str(p))
    except Exception as e: parse_errors.append((p.as_posix(),repr(e))); continue
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            for tgt in resolve(p,n):
                if tgt!=p.as_posix() and not tgt.endswith('/__init__.py') and p.name!='__init__.py': edges.add((p.as_posix(),tgt))
actual_nodes={p.as_posix() for p in py_files if p.name!='__init__.py'}
html=pathlib.Path('docs/dependency_graph.html').read_text()
html_nodes=set(re.findall(r'id:\s*"([^"]+\.py)"', html))
html_edges=set(re.findall(r'\{\s*source:\s*"([^"]+\.py)"\s*,\s*target:\s*"([^"]+\.py)"', html))
summary={
 'actual_nodes':len(actual_nodes),'html_nodes':len(html_nodes), 'missing_nodes':sorted(actual_nodes-html_nodes), 'extra_nodes':sorted(html_nodes-actual_nodes),
 'actual_edges':len(edges), 'html_edges':len(html_edges), 'missing_edges':sorted(edges-html_edges), 'extra_edges':sorted(html_edges-edges), 'parse_errors':parse_errors}
print(json.dumps(summary, indent=2))
raise SystemExit(0 if not summary['missing_nodes'] and not summary['extra_nodes'] and not summary['missing_edges'] and not summary['extra_edges'] and not parse_errors else 1)
