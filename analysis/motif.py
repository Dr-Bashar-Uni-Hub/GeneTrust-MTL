# NOTE: post hoc analysis script as run in the author's environment. Input paths (e.g. /tmp/...) are
# absolute; edit them to point to your copies of the files in data/ and results/ before running.
import numpy as np, pandas as pd, re
from scipy.stats import wilcoxon, mannwhitneyu
d=np.load('/mnt/user-data/uploads/Cell05_IG_structuredRNA.npz',allow_pickle=True)
df=pd.read_pickle('/tmp/all.pkl').drop_duplicates('NCBIGeneID').set_index('NCBIGeneID')
ids=d['gene_ids']; seqs=df.loc[ids].seq.values; cls=d['gt_classes'][d['y_true']]
ntok=(d['tokens']!=0).sum(1); assert all(n==min(len(s)-5,512) for n,s in zip(ntok,seqs)), 'token/sequence mismatch'
def nuc_att(a,s):                       # nucleotide score = mean of the tokens covering it
    n=min(len(s),517); T=min(len(s)-5,512); v=np.zeros(n); c=np.zeros(n)
    for i in range(T): v[i:i+6]+=a[i]; c[i:i+6]+=1
    v=v/np.maximum(c,1); return v/np.abs(v).sum()   # per-sequence normalisation
def find(seq,pat,lo,hi):
    m=[x.start() for x in re.finditer(f'(?=({pat}))',seq) if lo<=x.start()<=hi]; return m[0] if m else -1
MOT={'tRNA A box':('tRNA','T[AG]G[CT]..A[AG]..G',11,lambda s:(0,30)),
     'tRNA B box':('tRNA','GTTC[AG]A.[CT]C',9,lambda s:(40,len(s))),
     'snoRNA C box':('snoRNA','[AG]TGATGA',7,lambda s:(0,20)),
     'snoRNA D box':('snoRNA','CTGA',4,lambda s:(len(s)-20,len(s)-4))}
pos={m:np.array([find(s,p,*r(s)) if c==k else -1 for s,c in zip(seqs,cls)]) for m,(k,p,w,r) in MOT.items()}
tags=[k[5:] for k in d.files if k.startswith('att__')]
def relerr(t):
    a,p,b=d['att__'+t],d['prob__'+t],d['base__'+t]; dp=p-b
    return np.abs(a.sum(1)-dp)/np.maximum(np.abs(dp),1e-3)
NA={t:[nuc_att(d['att__'+t][i],s) for i,s in enumerate(seqs)] for t in tags}
groups={'Order-aware (PE), 5 seeds':[t for t in tags if '_PE_s' in t],'Order-blind (original), 5 seeds':[t for t in tags if t.startswith('MTL_Transformer_s')],'Untrained PE control':['MTL_Transformer_PE_untrained']}
rng=np.random.default_rng(0); rows=[]
for mname,(k,p,w,r) in MOT.items():
    has=pos[mname]>=0; idx=np.where(has)[0]
    for g,tl in groups.items():
        enr_seed=[]; pct_all=[]
        for t in tl:
            enr=[];pct=[]
            ok=relerr(t)<=0.10
            for i in idx:
                v=NA[t][i]; st=pos[mname][i]; inside=v[st:st+w].mean()
                null=[v[j:j+w].mean() for j in rng.integers(0,len(v)-w+1,200)]
                enr.append((inside-np.mean(null))*len(v))   # x len: units of "average nucleotide share"
                pct.append((np.array(null)<inside).mean())
            enr_seed.append(np.array(enr)); pct_all.append(np.array(pct))
        E=np.mean(enr_seed,0); Pc=np.mean(pct_all,0)
        st=wilcoxon(E).pvalue
        rows.append({'motif':mname,'n':len(idx),'model':g,'median_enrichment':np.median(E),'frac_seq_positive':(E>0).mean(),
                     'median_percentile_vs_random':np.median(Pc),'wilcoxon_p':st,
                     'seeds_with_positive_median':int(sum(np.median(e)>0 for e in enr_seed)),'n_seeds':len(tl)})
R=pd.DataFrame(rows); pd.set_option('display.width',220)
print(R.round(4).to_string(index=False))
R.to_csv('/mnt/user-data/outputs/analysis/motif_enrichment.csv',index=False)
# position-matched control (snoRNA only: motif-absent snoRNAs at same offsets)
print('\nPosition-matched control (attribution at same offset in same-class sequences WITHOUT the motif):')
pc=[]
for mname in ['snoRNA C box','snoRNA D box']:
    k,p,w,r=MOT[mname]; has=np.where(pos[mname]>=0)[0]; lack=np.where((cls=='snoRNA')&(pos[mname]<0))[0]
    for g,tl in groups.items():
        def score(i,off,from3):
            v=np.mean([NA[t][i] for t in tl],0); st=len(v)-off if from3 else off
            st=max(0,min(st,len(v)-w)); return (v[st:st+w].mean()-v.mean())*len(v)
        from3=mname.endswith('D box')
        offs=[ (len(seqs[i])-pos[mname][i]) if from3 else pos[mname][i] for i in has]
        a=[score(i,o,from3) for i,o in zip(has,offs)]
        b=[score(i,o,from3) for i in lack for o in [rng.choice(offs)]]
        u=mannwhitneyu(a,b,alternative='greater').pvalue
        pc.append({'motif':mname,'model':g,'n_with':len(a),'n_without':len(b),'median_with':np.median(a),'median_without':np.median(b),'MWU_p_greater':u})
PC=pd.DataFrame(pc); print(PC.round(4).to_string(index=False)); PC.to_csv('/mnt/user-data/outputs/analysis/motif_position_matched.csv',index=False)
# trained vs untrained, paired
print('\nTrained PE (seed-mean) vs untrained, paired Wilcoxon on enrichment:')
for mname,(k,p,w,r) in MOT.items():
    idx=np.where(pos[mname]>=0)[0]
    def e(tl): 
        return np.array([np.mean([(NA[t][i][pos[mname][i]:pos[mname][i]+w].mean()-NA[t][i].mean())*len(NA[t][i]) for t in tl]) for i in idx])
    a=e(groups['Order-aware (PE), 5 seeds']); b=e(groups['Untrained PE control'])
    print(f"  {mname:13s} median diff {np.median(a-b):.3f}  p={wilcoxon(a-b,alternative='greater').pvalue:.2e}")
