import numpy as np, pandas as pd, pickle, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
exec(open('/tmp/stats.py').read().split("M={'T'")[0].replace("print('Sharpr","pass #"))
D='/tmp/res/Results/Genomic_Transformer_Checkpoint_20260913_2127/data'
raw=pd.concat([pd.read_csv(f'{D}/train.csv'),pd.read_csv(f'{D}/test.csv')]).drop_duplicates('NCBIGeneID')
sp_=pd.read_csv('split_homology80.csv'); M_=raw.merge(sp_[['NCBIGeneID','split']],on='NCBIGeneID')
assert (M_[M_.split=='test'].NCBIGeneID.values==L['gene_ids']).all()
vid=M_[M_.split=='val'].NCBIGeneID.values
mfe=pickle.load(open('/mnt/user-data/outputs/analysis/mfe200.pkl','rb'))
df=pd.read_pickle('/tmp/all.pkl').drop_duplicates('NCBIGeneID').set_index('NCBIGeneID')
def feats(ids):
    d=df.loc[ids]; w=d.seq.str[:200]; m=np.array([mfe[g] for g in ids])
    gc=w.map(lambda q:(q.count('G')+q.count('C'))/max(len(q),1)).values
    return np.c_[np.log1p(np.minimum(d.L.values,517)),gc,m[:,0]/m[:,1],m[:,2]]
Fv,Ft=feats(vid),feats(L['gene_ids'])
cls=list(L['gt_classes']); yv=np.array([cls.index(c) for c in df.loc[vid].GeneType])
best=json.load(open('Cell03_best_lr.json'))
stack_corr=[]; pe_corr=[]
for sd in seeds:
    z=np.load(f"runs/MTL_Transformer_PE_lr{best['MTL_Transformer_PE']:g}_s{sd}/preds.npz")
    pv,pt=z['val_gene_type'],z['test_gene_type']
    Xv=np.c_[np.log(pv+1e-6),Fv]; Xt=np.c_[np.log(pt+1e-6),Ft]
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=5000)).fit(Xv,yv)
    yp=clf.classes_[clf.predict_proba(Xt).argmax(1)]
    stack_corr.append(yp==gt); pe_corr.append(pt.argmax(1)==gt)
sc=np.mean(stack_corr,0); pc=np.mean(pe_corr,0)
from sklearn.metrics import f1_score
labs=np.unique(gt)
print('PE alone acc %.4f | PE + biophysical (stacked) acc %.4f'%(pc.mean(),sc.mean()))
for sub,mask in [('all',None),('non-MPRA',~mpra)]:
    print(sub,'stacked - PE: %.4f [%.4f, %.4f] p=%.3f'%ci(sc,pc,mask))
S=pickle.load(open('/mnt/user-data/outputs/analysis/struct_preds.pkl','rb'))['preds']
print('stacked - structure GBM: %.4f [%.4f, %.4f] p=%.3f'%ci(sc,(S['Length+GC+Structure']==gt).astype(float)))
# per-class F1 of stacked (seed-averaged)
st_pred=[]
for sd in seeds:
    z=np.load(f"runs/MTL_Transformer_PE_lr{best['MTL_Transformer_PE']:g}_s{sd}/preds.npz")
    Xv=np.c_[np.log(z['val_gene_type']+1e-6),Fv]; Xt=np.c_[np.log(z['test_gene_type']+1e-6),Ft]
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=5000)).fit(Xv,yv)
    st_pred.append(clf.classes_[clf.predict_proba(Xt).argmax(1)])
f1=np.mean([f1_score(gt,p,labels=labs,average=None) for p in st_pred],0)
mf=np.mean([f1_score(gt,p,labels=labs,average='macro') for p in st_pred])
print('stacked macro-F1 %.4f'%mf); print(dict(zip(np.array(cls)[labs],f1.round(3))))
pickle.dump({'stack_pred':st_pred},open('/mnt/user-data/outputs/analysis/stack_preds.pkl','wb'))
