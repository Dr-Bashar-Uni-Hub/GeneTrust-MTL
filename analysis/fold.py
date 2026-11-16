# NOTE: post hoc analysis script as run in the author's environment. Input paths (e.g. /tmp/...) are
# absolute; edit them to point to your copies of the files in data/ and results/ before running.
import RNA, pandas as pd, numpy as np, pickle, os, time
out='/mnt/user-data/outputs/analysis/mfe200.pkl'
df=pd.read_pickle('/tmp/all.pkl').drop_duplicates('NCBIGeneID')
s=pd.read_csv('/tmp/jr/split_homology80.csv'); ids=s[s.split.isin(['train','test'])].NCBIGeneID
df=df.set_index('NCBIGeneID').loc[ids]
done=pickle.load(open(out,'rb')) if os.path.exists(out) else {}
t=time.time(); n=0
for g,q in zip(df.index,df.seq):
    if g in done: continue
    w=q[:200].replace('T','U'); ss,mfe=RNA.fold(w)
    done[g]=(mfe,len(w),ss.count('(')*2/len(w))
    n+=1
    if time.time()-t>270: break
pickle.dump(done,open(out,'wb')); print('folded now',n,'total',len(done),'of',len(df))
