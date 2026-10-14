"""Median survey numbers per kind of set.

    python tools/set_study/by_group.py patterns.csv
"""
import sys, os, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebrickable import sets
df=pd.read_csv(sys.argv[1])
df['set_num']=df.model.str.extract(r'^(\d+-\d+)')[0]
s=sets.set_index('set_num')
df=df.join(s[['name','year','path']],on='set_num')
def grp(p):
    if not isinstance(p,str): return 'other'
    if p.startswith(('Modular Buildings','Creator > Creator Expert')): return 'buildings_large'
    if p.startswith(('Town','City','Creator > Creator 3-in-1','Friends','Castle','Pirates','Western')): return 'town/creator'
    if p.startswith('Architecture'): return 'architecture'
    if p.startswith(('Technic','Bionicle','Mindstorms')): return 'technic'
    if p.startswith(('Brickheadz','Seasonal','Botanicals')): return 'sculpture/botanical'
    if p.startswith(('Star Wars','Speed Champions','Space','Train','Racers')): return 'vehicles'
    return 'other'
df['group']=df.path.map(grp)
cols=['parts','upright','inverted','sideways','angled','fill','shell_1','shell_2','bond','bricks','plates','tiles','slopes_up','slopes_inv','angled_plates','submodels','parts_per_step']
g=df[df.parts>=100].groupby('group')
out=g[cols].median().round(3); out.insert(0,'n',g.size())
print(out.T.to_string())
