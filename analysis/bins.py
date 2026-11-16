# NOTE: post hoc analysis script as run in the author's environment. Input paths (e.g. /tmp/...) are
# absolute; edit them to point to your copies of the files in data/ and results/ before running.
import numpy as np, pandas as pd, pickle
from sklearn.metrics import f1_score
exec(open('/tmp/stats.py').read().split("M={'T'")[0].replace("print('Sharpr","pass #"))
maxid=np.abs(L['maxid'])
bins=pd.cut(maxid,[-0.01,1e-4,0.5,0.65,0.8],labels=['none','<=0.50','0.50-0.65','0.65-0.80'])
rows=[]
for nm in ['MTL_Transformer_PE','MTL_Transformer','MTL_BiLSTM','STL_GeneType','Length-only GBM','1-NN 6-mer']:
    c=corr(nm,'gt'); r={'model':nm}
    for b in bins.categories: r[f'{b} (n={int((bins==b).sum())})']=round(c[bins==b].mean(),3)
    rows.append(r)
print('Gene Type accuracy by identity to nearest training sequence (identity x coverage)')
print(pd.DataFrame(rows).to_string(index=False))
# per-class F1 GT: PE (avg over seeds) vs Length-only
cls=L['gt_classes']; labs=np.unique(gt)
pe=np.mean([f1_score(gt,P[('MTL_Transformer_PE',sd,'gt')],labels=labs,average=None,zero_division=0) for sd in seeds],0)
t=np.mean([f1_score(gt,P[('MTL_Transformer',sd,'gt')],labels=labs,average=None,zero_division=0) for sd in seeds],0)
ln=f1_score(gt,C['Length-only GBM']['gt'],labels=labs,average=None,zero_division=0)
print(pd.DataFrame({'class':cls[labs],'n_test':[int((gt==l).sum()) for l in labs],'F1_PE':pe.round(3),'F1_T':t.round(3),'F1_Length':ln.round(3)}).to_string(index=False))
# MPRA subset accuracy
for nm in ['MTL_Transformer_PE','Length-only GBM']: print(nm,'acc on MPRA tiles',corr(nm,'gt')[mpra].mean().round(4))
