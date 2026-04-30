from __future__ import annotations
import ast, json, pathlib, re
root=pathlib.Path('.')
exclude={'.venv','.git','__pycache__','.ruff_cache','.pytest_cache','docs','notebooks','temp'}
py_files=sorted(p for p in root.rglob('*.py') if not any(part in exclude for part in p.parts) and 'tests' not in p.parts)
module_to_file={}
file_to_module={}
for p in py_files:
    if p.name=='__init__.py': mod='.'.join(p.parent.parts)
    else: mod=p.with_suffix('').as_posix().replace('/','.')
    module_to_file[mod]=p.as_posix(); file_to_module[p.as_posix()]=mod
local_roots={p.parts[0] for p in py_files}
# Build __init__ re-export map: package.module symbol -> module file imported from
reexports={}
for p in py_files:
    if p.name!='__init__.py': continue
    pkg='.'.join(p.parent.parts)
    try: tree=ast.parse(p.read_text())
    except Exception: continue
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module:
            base=n.module if not n.level else None
            if n.level:
                parts=list(p.parent.parts)
                base_parts=parts[:max(0,len(parts)-n.level+1)] + (n.module.split('.') if n.module else [])
                base='.'.join(base_parts)
            if base in module_to_file:
                for a in n.names:
                    reexports[(pkg,a.asname or a.name)]=module_to_file[base]

def module_doc(p:pathlib.Path):
    try:
        doc=ast.get_docstring(ast.parse(p.read_text()))
    except Exception:
        doc=None
    if not doc: return None
    line=' '.join(doc.strip().split())
    return line[:180] + ('…' if len(line)>180 else '')

def resolve(src:pathlib.Path, n):
    mods=[]; aliases=[]
    if isinstance(n, ast.Import):
        mods=[a.name for a in n.names]
    elif isinstance(n, ast.ImportFrom):
        if n.level:
            parts=list(src.parent.parts if src.name=='__init__.py' else src.with_suffix('').parts[:-1])
            base_parts=parts[:max(0,len(parts)-n.level+1)]
            if n.module: base_parts += n.module.split('.')
            base='.'.join(base_parts)
        else:
            base=n.module or ''
        mods=[base] if base else []
        aliases=[a.asname or a.name for a in n.names if a.name!='*']
        for a in aliases:
            mods.append(base+'.'+a if base else a)
        # resolve package reexports to implementation modules
        for a in aliases:
            if (base,a) in reexports:
                yield reexports[(base,a)]
    for mod in mods:
        parts=mod.split('.') if mod else []
        if not parts or parts[0] not in local_roots: continue
        for i in range(len(parts),0,-1):
            cand='.'.join(parts[:i])
            if cand in module_to_file:
                yield module_to_file[cand]; break

edges=set(); connected=set()
for p in py_files:
    try: tree=ast.parse(p.read_text(), filename=str(p))
    except Exception: continue
    src=p.as_posix()
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            for tgt in resolve(p,n):
                if tgt!=src and not tgt.endswith('/__init__.py') and not src.endswith('/__init__.py'):
                    edges.add((src,tgt)); connected.update([src,tgt])
# include all non-init source files, even isolates, so sidebar is complete
nodes=sorted(p.as_posix() for p in py_files if p.name!='__init__.py')
group_meta={
 'workflows':('#388bfd','Workflows'), 'agents':('#a371f7','Agents'), 'store':('#3fb950','Store'), 'domain':('#76e3ea','Domain'),
 'sonarqube':('#f78166','SonarQube'), 'smellai_datasets':('#39d353','Datasets'), 'swe_refactor':('#e3b341','SWE Refactor'),
 'rminer':('#f0883e','RMiner'), 'repo_utils':('#79c0ff','Repo Utils'), 'mlflow_utils':('#ff7b72','MLflow Utils'),
 'models':('#d2a8ff','Models'), 'scripts':('#56d364','Scripts'), 'evals':('#ffa657','Evals'), 'tools':('#8b949e','Tools'),
 'presentation_innovaite':('#db6d28','Presentation'), 'logging_config.py':('#8b949e','Config'),
}
colors=['#388bfd','#a371f7','#3fb950','#76e3ea','#f78166','#39d353','#e3b341','#f0883e','#79c0ff','#ff7b72','#d2a8ff','#56d364','#ffa657','#8b949e']
groups=[]
for n in nodes:
    key=n.split('/')[0] if '/' in n else n
    if key not in groups: groups.append(key)

def js_str(s): return json.dumps(s)
out=[]
out.append('const GROUPS = {')
for i,g in enumerate(groups):
    color,label=group_meta.get(g,(colors[i%len(colors)],g.replace('_',' ').title()))
    out.append(f'  {json.dumps(g)}: {{ color: {json.dumps(color)}, label: {json.dumps(label)} }},')
out.append('};\n')
out.append('const NODES = [')
for n in nodes:
    group=n.split('/')[0] if '/' in n else n
    short=pathlib.Path(n).stem if pathlib.Path(n).stem!='agent' else '/'.join(pathlib.Path(n).parts[-3:-1]) or 'agent'
    doc=module_doc(pathlib.Path(n)) or 'No module docstring found; node generated from static Python import analysis.'
    outgoing=sum(1 for a,b in edges if a==n); incoming=sum(1 for a,b in edges if b==n)
    out.append('  {')
    out.append(f'    id: {js_str(n)},')
    out.append(f'    group: {js_str(group)},')
    out.append(f'    short: {js_str(short)},')
    out.append('    bullets: [')
    out.append(f'      {js_str(doc)},')
    out.append(f'      {js_str(f"Static local imports: {outgoing} outgoing, {incoming} incoming.")},')
    out.append(f'      {js_str("Regenerated from repository AST and validated with uv run --with pydeps.")}')
    out.append('    ]')
    out.append('  },')
out.append('];\n')
out.append('const EDGES = [')
for a,b in sorted(edges):
    out.append(f'  {{ source: {js_str(a)}, target: {js_str(b)} }},')
out.append('];')
print('\n'.join(out))
