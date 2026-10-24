import numpy as np, pandas as pd, pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
mfe=pickle.load(open('/mnt/user-data/outputs/analysis/mfe200.pkl','rb'))
df=pd.read_pickle('/tmp/all.pkl').drop_duplicates('NCBIGeneID').set_index('NCBIGeneID')
s=pd.read_csv('split_homology80.csv'); L=np.load('split_labels_test.npz',allow_pickle=True)
def feats(ids):
    d=df.loc[ids]; w=d.seq.str[:200]
    m=np.array([mfe[g] for g in ids])
    gc=w.map(lambda q:(q.count('G')+q.count('C'))/max(len(q),1)).values
    return pd.DataFrame({'loglen':np.log1p(np.minimum(d.L.values,517)),'gc200':gc,'mfe_per_nt':m[:,0]/m[:,1],'paired_frac':m[:,2]})
trid=s[s.split=='train'].NCBIGeneID.values; teid=L['gene_ids']
Ftr,Fte=feats(trid),feats(teid)
le=LabelEncoder().fit(df.loc[trid].GeneType); ytr=le.transform(df.loc[trid].GeneType); gt=L['gt']; assert (le.classes_==L['gt_classes']).all()
labs=np.unique(gt); preds={}
for name,cols in [('Length','loglen'.split()),('Length+GC',['loglen','gc200']),('Length+GC+Structure',['loglen','gc200','mfe_per_nt','paired_frac'])]:
    p=HistGradientBoostingClassifier(random_state=0).fit(Ftr[cols],ytr).predict(Fte[cols]); preds[name]=p
    print(f"{name:22s} GT acc {accuracy_score(gt,p):.4f}  macroF1 {f1_score(gt,p,labels=labs,average='macro'):.4f}")
P=pickle.load(open('/tmp/P.pkl','rb'))
pe=np.mean([f1_score(gt,P[('MTL_Transformer_PE',sd,'gt')],labels=labs,average=None) for sd in [1,7,21,42,100]],0)
tab=pd.DataFrame({'class':le.classes_[labs],'n':[int((gt==l).sum()) for l in labs]})
for k,p in preds.items(): tab[k]=f1_score(gt,p,labels=labs,average=None).round(3)
tab['PE Transformer']=pe.round(3); print(tab.to_string(index=False))
pickle.dump({'preds':preds,'Fte':Fte},open('/mnt/user-data/outputs/analysis/struct_preds.pkl','wb'))
# class-wise feature summaries (test+train), to describe biology
allF=pd.concat([Ftr.assign(c=df.loc[trid].GeneType.values),Fte.assign(c=le.classes_[gt])])
print(allF.groupby('c')[['gc200','mfe_per_nt','paired_frac']].median().round(3))
