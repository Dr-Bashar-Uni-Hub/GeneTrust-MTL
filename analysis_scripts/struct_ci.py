import numpy as np, pandas as pd, pickle
exec(open('/tmp/stats.py').read().split("M={'T'")[0].replace("print('Sharpr","pass #"))
S=pickle.load(open('/mnt/user-data/outputs/analysis/struct_preds.pkl','rb'))['preds']
st=(S['Length+GC+Structure']==gt).astype(float)
out=[]
for nm in ['MTL_Transformer_PE','MTL_Transformer']:
    for sub,mask in [('all',None),('non-MPRA',~mpra)]:
        e,lo,hi,p=ci(corr(nm,'gt'),st,mask); out.append((nm,'- Length+GC+Structure GBM',sub,round(e,4),round(lo,4),round(hi,4),round(p,3)))
e,lo,hi,p=ci(st,corr('Length-only GBM','gt')); out.append(('Length+GC+Structure GBM','- Length-only GBM','all',round(e,4),round(lo,4),round(hi,4),round(p,3)))
for o in out: print(o)
pd.DataFrame(out,columns=['a','b','subset','diff','lo','hi','p']).to_csv('/mnt/user-data/outputs/analysis/structure_contrasts.csv',index=False)
