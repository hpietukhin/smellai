from __future__ import annotations
import ast, json, pathlib, re
# get actual nodes/edges using prior generator by importing? keep standalone enough
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
            base='.'.join(list(p.parent.parts)[:max(0,len(p.parent.parts)-n.level+1)] + n.module.split('.')) if n.level else n.module
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

def module_doc(p:pathlib.Path):
    try: doc=ast.get_docstring(ast.parse(p.read_text()))
    except Exception: doc=None
    if not doc: return 'No module docstring found; node generated from static Python import analysis.'
    line=' '.join(doc.strip().split())
    return line[:180]+('…' if len(line)>180 else '')

edges=set()
for p in py_files:
    if p.name=='__init__.py': continue
    try: tree=ast.parse(p.read_text(), filename=str(p))
    except Exception: continue
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            for tgt in resolve(p,n):
                if tgt!=p.as_posix() and not tgt.endswith('/__init__.py'): edges.add((p.as_posix(),tgt))
actual_nodes=sorted(p.as_posix() for p in py_files if p.name!='__init__.py')
# parse old node object blocks
old_html=pathlib.Path('temp/dependency_validation/original_dependency_graph.html').read_text()
start=old_html.index('const NODES = [')+len('const NODES = [')
end=old_html.index('];', start)
arr=old_html[start:end]
old_objects={}; old_order=[]
i=0
while i < len(arr):
    if arr[i]=='{':
        depth=0; j=i; in_str=False; esc=False
        while j < len(arr):
            c=arr[j]
            if in_str:
                if esc: esc=False
                elif c=='\\': esc=True
                elif c=='"': in_str=False
            else:
                if c=='"': in_str=True
                elif c=='{': depth+=1
                elif c=='}':
                    depth-=1
                    if depth==0:
                        block=arr[i:j+1]
                        m=re.search(r'id:\s*"([^"]+)"', block)
                        if m:
                            old_objects[m.group(1)]=block.strip(); old_order.append(m.group(1))
                        i=j
                        break
            j+=1
    i+=1
# groups
base_groups=[('workflows','#388bfd','Workflows'),('agents','#a371f7','Agents'),('store','#3fb950','Store'),('domain','#76e3ea','Domain'),('sonarqube','#f78166','SonarQube'),('smellai_datasets','#39d353','Datasets'),('swe_refactor','#e3b341','SWE Refactor'),('rminer','#f0883e','RMiner'),('repo_utils','#79c0ff','Repo Utils'),('mlflow_utils','#ff7b72','MLflow Utils'),('models','#d2a8ff','Models'),('scripts','#56d364','Scripts'),('evals','#ffa657','Evals'),('tools','#8b949e','Tools'),('presentation_innovaite','#db6d28','Presentation'),('logging_config.py','#8b949e','Config')]
used=[]
for n in actual_nodes:
    g=n.split('/')[0] if '/' in n else n
    if g not in used: used.append(g)
meta={g:(c,l) for g,c,l in base_groups}
node_order=[n for n in old_order if n in actual_nodes] + [n for n in actual_nodes if n not in old_objects]
# append any old nodes not actual? none expected excluded

def js(s): return json.dumps(s)
out=[]
out.append('const GROUPS = {')
for g,c,l in base_groups:
    if g in used: out.append(f'  {g}:'.ljust(22)+f'{{ color: {js(c)}, label: {js(l)} }},')
out.append('};\n')
out.append('const NODES = [')
for n in node_order:
    if n in old_objects:
        out.append(old_objects[n]+',')
    else:
        group=n.split('/')[0] if '/' in n else n
        stem=pathlib.Path(n).stem
        short=stem if stem!='agent' else '/'.join(pathlib.Path(n).parts[-3:-1])
        outgoing=sum(1 for a,b in edges if a==n); incoming=sum(1 for a,b in edges if b==n)
        out.append('  {')
        out.append(f'    id: {js(n)},')
        out.append(f'    group: {js(group)},')
        out.append(f'    short: {js(short)},')
        out.append('    bullets: [')
        out.append(f'      {js(module_doc(pathlib.Path(n)))},')
        out.append(f'      {js(f"Static local imports: {outgoing} outgoing, {incoming} incoming.")},')
        out.append(f'      {js("Added during dependency graph renewal; generated from repository AST and validated with uv run --with pydeps.")}')
        out.append('    ]')
        out.append('  },')
out.append('];\n')
out.append('const EDGES = [')
# group edge comments omitted; validated exact static edge list
for a,b in sorted(edges): out.append(f'  {{ source: {js(a)}, target: {js(b)} }},')
out.append('];')
print('\n'.join(out))
