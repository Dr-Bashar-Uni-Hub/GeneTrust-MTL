# NOTE: post hoc analysis script as run in the author's environment. Input paths (e.g. /tmp/...) are
# absolute; edit them to point to your copies of the files in data/ and results/ before running.
import numpy as np, pandas as pd, pickle, json, itertools, scipy.sparse as sp
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
Z=np.load('Cell11_mouse_predictions.npz',allow_pickle=True); cls=list(Z['classes']); y=Z['y_true']; ids=Z['gene_ids']
M=pd.read_csv('mouse_external.csv').set_index('NCBIGeneID').loc[ids]
mx=M.maxid_to_human_train.values; comp=M.mouse_component.values
s80=mx<0.8; s60=mx<0.6
# human training data and features
df=pd.read_pickle('/tmp/all.pkl').drop_duplicates('NCBIGeneID').set_index('NCBIGeneID')
spl=pd.read_csv('/tmp/jr/split_homology80.csv'); trid=spl[spl.split=='train'].NCBIGeneID.values
D='/tmp/rev/Results/Genomic_Transformer_Checkpoint_20260913_2127/data'
raw=pd.concat([pd.read_csv(f'{D}/train.csv'),pd.read_csv(f'{D}/test.csv')]).drop_duplicates('NCBIGeneID')
M_=raw.merge(spl[['NCBIGeneID','split']],on='NCBIGeneID'); vid=M_[M_.split=='val'].NCBIGeneID.values
mh=pickle.load(open('/mnt/user-data/outputs/analysis/mfe200.pkl','rb')); mm=pickle.load(open('/mnt/user-data/outputs/analysis/mfe200_mouse.pkl','rb'))
def feats(seqs,keys,mf):
    w=pd.Series(seqs).str[:200]; m=np.array([mf[k] for k in keys]); L=pd.Series(seqs).str.len().values
    return np.c_[np.log1p(np.minimum(L,517)),w.map(lambda q:(q.count('G')+q.count('C'))/len(q)).values,m[:,0]/m[:,1],m[:,2]]
Ftr=feats(df.loc[trid].seq.values,trid,mh); Fv=feats(df.loc[vid].seq.values,vid,mh); Fm=feats(M.seq.values,ids,mm)
ytr=np.array([cls.index(c) for c in df.loc[trid].GeneType]); yv=np.array([cls.index(c) for c in df.loc[vid].GeneType])
C={}
G=lambda X,Xt: HistGradientBoostingClassifier(random_state=0).fit(X,ytr).predict(Xt)
C['Length only']=[G(Ftr[:,:1],Fm[:,:1])]; C['Length+GC+structure']=[G(Ftr,Fm)]
idx={''.join(p):i for i,p in enumerate(itertools.product('ACGT',repeat=6))}
def K(seqs):
    r,c=[],[]
    for i,q in enumerate(seqs):
        q=q[:517]
        for j in range(len(q)-5): r.append(i);c.append(idx[q[j:j+6]])
    return normalize(sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(len(seqs),4096)))
ni=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute').fit(K(df.loc[trid].seq.values)).kneighbors(K(M.seq.values))[1][:,0]
C['1-NN lookup']=[ytr[ni]]
S5=[1,7,21,42,100]
for nm,lab in [('MTL_Transformer_PE','Transformer+PE'),('MTL_Transformer','Transformer (order-blind)'),('STL_GeneType','Single-task Transformer'),('MTL_BiLSTM','MTL BiLSTM'),('MTL_CNN','MTL CNN')]:
    C[lab]=[Z[f'prob__{nm}_s{s}'].argmax(1) for s in S5]
best=json.load(open('/tmp/jr/Cell03_best_lr.json'))
def comb(val_probs,mouse_probs):
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=5000)).fit(np.c_[np.log(val_probs+1e-6),Fv],yv)
    return clf.classes_[clf.predict_proba(np.c_[np.log(mouse_probs+1e-6),Fm]).argmax(1)]
C['Transformer+PE + biophysical']=[comb(np.load(f"/tmp/jr/runs/MTL_Transformer_PE_lr{best['MTL_Transformer_PE']:g}_s{s}/preds.npz")['val_gene_type'],Z[f'prob__MTL_Transformer_PE_s{s}']) for s in S5]
fm={s:np.load(f'runs_fm_ext/DNABERT6_GeneType_s{s}/preds.npz',allow_pickle=True) for s in [1,21,42]}
assert all((fm[s]['mouse_gene_ids']==ids).all() for s in fm)
C['DNABERT']=[fm[s]['mouse_gene_type'].argmax(1) for s in fm]
C['DNABERT + biophysical']=[comb(fm[s]['val_gene_type'],fm[s]['mouse_gene_type']) for s in fm]
corr={k:np.mean([p==y for p in v],0) for k,v in C.items()}
labs=np.unique(y)
def mF1(v,mask): return np.mean([f1_score(y[mask],p[mask],labels=np.unique(y[mask]),average='macro',zero_division=0) for p in v])
rows=[]
for k,v in C.items():
    rows.append([k,f'{corr[k].mean():.3f}',f'{mF1(v,np.ones(len(y),bool)):.3f}',f'{corr[k][s80].mean():.3f}',f'{corr[k][s60].mean():.3f}',f'{corr[k][mx<=1e-4].mean():.3f}'])
T=pd.DataFrame(rows,columns=['model','acc all','macroF1 all','acc no>=80% human match','acc no>=60%','acc no detectable homolog'])
print(f'n all {len(y)}, strict80 {s80.sum()}, strict60 {s60.sum()}, no homolog {(mx<=1e-4).sum()}'); print(T.to_string(index=False))
T.to_csv('/mnt/user-data/outputs/analysis/mouse_accuracy.csv',index=False)
uc=np.unique(comp); mem=[np.where(comp==c)[0] for c in uc]; rng=np.random.default_rng(0)
B=[np.concatenate([mem[j] for j in rng.integers(0,len(uc),len(uc))]) for _ in range(2000)]
def ci(a,b,mask):
    d=corr[a]-corr[b]; bs=np.array([d[ix[mask[ix]]].mean() for ix in B]); return d[mask].mean(),*np.percentile(bs,[2.5,97.5]),2*min((bs<=0).mean(),(bs>=0).mean())
out=[]
for a,b in [('Transformer+PE','Transformer (order-blind)'),('Transformer (order-blind)','Single-task Transformer'),('Transformer (order-blind)','Length only'),('Transformer+PE','Length only'),('Transformer+PE','Length+GC+structure'),('Length+GC+structure','Length only'),('Transformer+PE + biophysical','Transformer+PE'),('DNABERT','Transformer+PE'),('DNABERT + biophysical','DNABERT'),('Transformer+PE','MTL BiLSTM'),('Transformer+PE','1-NN lookup')]:
    for sub,mask in [('all',np.ones(len(y),bool)),('no>=60%',s60)]:
        e,lo,hi,p=ci(a,b,mask); out.append([f'{a} − {b}',sub,round(100*e,1),round(100*lo,1),round(100*hi,1),round(p,3)])
R=pd.DataFrame(out,columns=['contrast','subset','diff_pp','lo','hi','p']); print(R.to_string(index=False)); R.to_csv('/mnt/user-data/outputs/analysis/mouse_contrasts.csv',index=False)
f=lambda p: f1_score(y,p,labels=labs,average=None,zero_division=0)
PC=pd.DataFrame({'class':np.array(cls)[labs],'n':[int((y==l).sum()) for l in labs]})
for k in ['Length only','Length+GC+structure','Transformer (order-blind)','Transformer+PE','Transformer+PE + biophysical','DNABERT','DNABERT + biophysical']: PC[k]=np.mean([f(p) for p in C[k]],0).round(3)
print(PC.to_string(index=False)); PC.to_csv('/mnt/user-data/outputs/analysis/mouse_perclass.csv',index=False)
bio=M[M.GeneType=='BIOLOGICAL_REGION']; print('mouse BIOLOGICAL_REGION: top lengths',bio.L.value_counts().head(3).to_dict(),'| share of most common length %.2f'%(bio.L.value_counts().iloc[0]/len(bio)))
print('mouse BIO top descriptions',bio.Description.str[:40].value_counts().head(4).to_dict())
pickle.dump({'corr':corr,'C':C},open('/mnt/user-data/outputs/analysis/mouse_corr.pkl','wb'))
