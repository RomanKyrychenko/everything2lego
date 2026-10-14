"""Rebrickable's database dumps (https://rebrickable.com/downloads/) as
data frames, with set themes as paths and the model groups the survey
compares.  Put the .csv.gz files in $LEGO_DATA."""
import os
import pandas as pd
D=os.environ.get('LEGO_DATA','/Volumes/PRO-G40/lego_data')+'/'
th=pd.read_csv(D+'themes.csv.gz'); sets=pd.read_csv(D+'sets.csv.gz')
parts=pd.read_csv(D+'parts.csv.gz'); cats=pd.read_csv(D+'part_categories.csv.gz')
cols=pd.read_csv(D+'colors.csv.gz'); inv=pd.read_csv(D+'inventories.csv.gz')
ip=pd.read_csv(D+'inventory_parts.csv.gz', usecols=['inventory_id','part_num','color_id','quantity','is_spare'])
ip=ip[~ip.is_spare]
par=dict(zip(th.id,th.parent_id)); tname=dict(zip(th.id,th.name))
def path(t):
    p=[]
    while True:
        p.append(tname[t])
        if pd.isna(par.get(t)): break
        t=int(par[t])
    return ' > '.join(reversed(p))
sets['path']=sets.theme_id.map(path)
inv1=inv.sort_values('version').drop_duplicates('set_num')
GROUPS={
 'buildings': lambda p,n: p.startswith('Modular Buildings') or p.startswith('Creator > Creator Expert') or p in ('City','LEGO Ideas and CUUSOO','Icons','Creator > Creator 3-in-1','Harry Potter','Friends'),
 'sculpture': lambda p,n: p.startswith('Brickheadz') or p.startswith('Seasonal') ,
 'vehicles':  lambda p,n: p.startswith('Speed Champions') or p.startswith('City > Police') or p.startswith('City > Fire') or p.startswith('City > Traffic'),
 'architecture': lambda p,n: p.startswith('Architecture'),
 'botanical': lambda p,n: p.startswith('Botanicals'),
}
modern=sets[(sets.year>=2015)&(sets.num_parts>=150)]
def group_parts(g):
    s=modern[[GROUPS[g](p,n) for p,n in zip(modern.path,modern.name)]]
    i=inv1[inv1.set_num.isin(s.set_num)]
    x=ip[ip.inventory_id.isin(i.id)].merge(i[['id','set_num']],left_on='inventory_id',right_on='id')
    return s, x.merge(parts[['part_num','name','part_cat_id']],on='part_num',how='left')
