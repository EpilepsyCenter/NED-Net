import sqlite3, re, collections, os, json, numpy as np
from scipy import stats
P=os.path.expanduser('~/.eeg_seizure_analyzer/projects/sv2a_spikes_wk1-6.db')
c=sqlite3.connect(P); c.row_factory=sqlite3.Row
wk={}
for r in c.execute('select id, path from chunks'):
    m=re.search(r'Week(\d+)-Day(\d+)', r['path']); wk[r['id']]= int(m.group(1)) if m else None
EXCLUDE={'30','355676','372837','x'}
grp={}
for r in c.execute('select distinct animal_id, group_id from file_animals'):
    g=(r['group_id'] or '').strip().lower()
    if g in ('control','sv2a') and r['animal_id'] not in EXCLUDE: grp[r['animal_id']]=g
hrs=collections.Counter()
for r in c.execute('select animal_id a, chunk_id ci, valid_sec v from file_animals'):
    w=wk.get(r['ci'])
    if w is None or r['a'] not in grp: continue
    hrs[(r['a'],'base' if w<=3 else 'lev')] += (r['v'] or 0)/3600.0
n=collections.Counter(); dur=collections.defaultdict(float)
for r in c.execute('select animal_id a, chunk_id ci, duration_sec d from events where excluded=0'):
    w=wk.get(r['ci'])
    if w is None or r['a'] not in grp: continue
    k=(r['a'],'base' if w<=3 else 'lev'); n[k]+=1; dur[k]+=r['d'] or 0
rows=[]
print(f"{'animal':9}{'grp':8}{'ph':6}{'hours':9}{'spikes':9}{'spikes/h':10}{'meanDur_ms':10}")
for a in sorted(grp,key=lambda x:(grp[x],x)):
    for ph in ('base','lev'):
        h=hrs[(a,ph)]
        if h<=0: continue
        rate=n[(a,ph)]/h; md=1000*dur[(a,ph)]/max(n[(a,ph)],1)
        rows.append(dict(a=a,g=grp[a],ph=ph,h=h,n=n[(a,ph)],rate=rate,md=md))
        print(f"{a:9}{grp[a]:8}{ph:6}{h:<9.0f}{n[(a,ph)]:<9}{rate:<10.1f}{md:<10.1f}")
json.dump(rows, open('spikes_per_animal.json','w'), indent=1)
print()
def get(g,ph,k): return np.array([r[k] for r in rows if r['g']==g and r['ph']==ph], float)
for k,lab in [('rate','spikes/h'),('md','spike duration (ms)')]:
    a=get('control','base',k); b=get('sv2a','base',k)
    print(f'{lab:22} ctrl n={len(a)} med={np.median(a):.2f} | sv2a n={len(b)} med={np.median(b):.2f} | '
          f'Shapiro {stats.shapiro(a).pvalue:.4f}/{stats.shapiro(b).pvalue:.4f} | MWU p={stats.mannwhitneyu(a,b).pvalue:.4g}')
for g in ('control','sv2a'):
    base={r['a']:r['rate'] for r in rows if r['g']==g and r['ph']=='base'}
    lev={r['a']:r['rate'] for r in rows if r['g']==g and r['ph']=='lev'}
    ids=[x for x in base if x in lev]
    x=np.array([base[i] for i in ids]); y=np.array([lev[i] for i in ids])
    print(f'{g:8} LEV spikes/h n={len(ids)} base med={np.median(x):.1f} lev med={np.median(y):.1f} Wilcoxon p={stats.wilcoxon(x,y).pvalue:.4g}')
