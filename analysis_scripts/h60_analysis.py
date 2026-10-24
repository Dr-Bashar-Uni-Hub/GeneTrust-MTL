import numpy as np, pandas as pd, pickle, json, itertools, scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler, normalize
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
L=np.load('split60_labels_test.npz',allow_pickle=True); gt,sf,fr=L['gt'],L['sf'],L['fr'].astype(int); ids=L['gene_ids']
labs=np.unique(gt); cls=list(L['gt_classes'])
spl=pd.read_csv('split_homology60.csv').set_index('NCBIGeneID'); clu=spl.loc[ids,'cluster'].values
df=pd.read_pickle('/tmp/all.pkl').drop_duplicates('NCBIGeneID').set_index('NCBIGeneID')
mpra=(df.loc[ids].Description.str.contains('Sharpr-MPRA')&(df.loc[ids].L==295)).values
D='/tmp/rev/Results/Genomic_Transformer_Checkpoint_20260913_2127/data'
raw=pd.concat([pd.read_csv(f'{D}/train.csv'),pd.read_csv(f'{D}/test.csv')]).drop_duplicates('NCBIGeneID')
M_=raw.merge(spl.reset_index()[['NCBIGeneID','split']],on='NCBIGeneID'); assert (M_[M_.split=='test'].NCBIGeneID.values==ids).all()
trid=M_[M_.split=='train'].NCBIGeneID.values; vid=M_[M_.split=='val'].NCBIGeneID.values
mfe=pickle.load(open('/mnt/user-data/outputs/analysis/mfe200.pkl','rb'))
def feats(g):
    d=df.loc[g]; w=d.seq.str[:200]; m=np.array([mfe[x] for x in g])
    return np.c_[np.log1p(np.minimum(d.L.values,517)),w.map(lambda q:(q.count('G')+q.count('C'))/len(q)).values,m[:,0]/m[:,1],m[:,2]]
Ftr,Fv,Fte=feats(trid),feats(vid),feats(ids)
y_of=lambda g: np.array([cls.index(c) for c in df.loc[g].GeneType])
fam=lambda s: 'IMMUNOGLOBULIN' if s.startswith('IGH') else 'RNA_GENE' if s.startswith(('RNU','SNORD')) else 'LOCUS' if s.startswith('LOC') else 'OTHER'
sfc=list(L['sf_classes']); frc=list(L['fr_classes'])
ytr=y_of(trid); yv=y_of(vid)
sftr=np.array([sfc.index(fam(s)) for s in df.loc[trid].Symbol]); mlb=MultiLabelBinarizer(classes=frc).fit([frc])
frtr=mlb.transform(df.loc[trid].roles.str.split('|'))
corr={}
G=lambda X,y,Xt: HistGradientBoostingClassifier(random_state=0).fit(X,y).predict(Xt)
pl=G(Ftr[:,:1],ytr,Fte[:,:1]); corr[('LEN','gt')]=(pl==gt).astype(float)
corr[('LEN','sf')]=(G(Ftr[:,:1],sftr,Fte[:,:1])==sf).astype(float)
corr[('LEN','fr')]=(np.stack([G(Ftr[:,:1],frtr[:,j],Fte[:,:1]) for j in range(4)],1)==fr).all(1).astype(float)
ps=G(Ftr,ytr,Fte); corr[('STR','gt')]=(ps==gt).astype(float)
idx={''.join(p):i for i,p in enumerate(itertools.product('ACGT',repeat=6))}
def K(g):
    r,c=[],[]
    for i,q in enumerate(df.loc[g].seq):
        q=q[:517]
        for j in range(len(q)-5): r.append(i);c.append(idx[q[j:j+6]])
    return normalize(sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(len(g),4096)))
nn=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute').fit(K(trid)); ni=nn.kneighbors(K(ids))[1][:,0]
corr[('NN','gt')]=(ytr[ni]==gt).astype(float)
best=json.load(open('/tmp/jr/Cell03_best_lr.json')); seeds=[1,21,42]
P={}
for m in ['MTL_Transformer_PE','MTL_Transformer','STL_GeneType','STL_SymbolFamily','STL_FunctionalRoles']:
    for s in seeds:
        z=np.load(f"runs_h60/{m}_lr{best[m]:g}_s{s}/preds.npz"); P[(m,s)]=z
def seedcorr(m,t):
    key={'gt':'gene_type','sf':'symbol_family','fr':'functional_roles'}[t]; y={'gt':gt,'sf':sf,'fr':fr}[t]
    out=[]
    for s in seeds:
        p=P[(m,s)][f'test_{key}']; out.append(((p>=.5).astype(int)==y).all(1) if t=='fr' else p.argmax(1)==y)
    return np.mean(out,0)
for t in ['gt','sf','fr']:
    corr[('PE',t)]=seedcorr('MTL_Transformer_PE',t); corr[('T',t)]=seedcorr('MTL_Transformer',t)
corr[('STL','gt')]=seedcorr('STL_GeneType','gt'); corr[('STL','sf')]=seedcorr('STL_SymbolFamily','sf'); corr[('STL','fr')]=seedcorr('STL_FunctionalRoles','fr')
# combined
cp=[]; cpred=[]
for s in seeds:
    z=P[('MTL_Transformer_PE',s)]
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=5000)).fit(np.c_[np.log(z['val_gene_type']+1e-6),Fv],yv)
    pr=clf.classes_[clf.predict_proba(np.c_[np.log(z['test_gene_type']+1e-6),Fte]).argmax(1)]; cpred.append(pr); cp.append(pr==gt)
corr[('COMB','gt')]=np.mean(cp,0)
uc=np.unique(clu); mem=[np.where(clu==c)[0] for c in uc]; rng=np.random.default_rng(0)
B=[np.concatenate([mem[j] for j in rng.integers(0,len(uc),len(uc))]) for _ in range(2000)]
def ci(a,b,mask=None):
    m=np.ones(len(gt),bool) if mask is None else mask; d=a-b; bs=np.array([d[ix[m[ix]]].mean() for ix in B])
    return d[m].mean(),*np.percentile(bs,[2.5,97.5]),2*min((bs<=0).mean(),(bs>=0).mean())
print('ACCURACY (60pct split, n=',len(gt),'MPRA tiles',mpra.sum())
for k,v in sorted(corr.items()): print(f'  {k[0]:5s} {k[1]}: all {v.mean():.4f}  non-MPRA {v[~mpra].mean():.4f}')
rows=[]
for t,a,b in [('gt','T','STL'),('sf','T','STL'),('fr','T','STL'),('gt','PE','T'),('sf','PE','T'),('fr','PE','T'),('gt','T','LEN'),('gt','PE','LEN'),('sf','PE','LEN'),('fr','PE','LEN'),('gt','PE','STR'),('gt','STR','LEN'),('gt','COMB','PE'),('gt','COMB','STR')]:
    for sub,mk in [('all',None),('nonMPRA',~mpra)]:
        e,lo,hi,p=ci(corr[(a,t)],corr[(b,t)],mk); rows.append((t,f'{a}-{b}',sub,round(100*e,1),round(100*lo,1),round(100*hi,1),round(p,3)))
R=pd.DataFrame(rows,columns=['task','contrast','subset','diff_pp','lo','hi','p']); print(R.to_string(index=False))
R.to_csv('/mnt/user-data/outputs/analysis/h60_contrasts.csv',index=False)
f=lambda p: f1_score(gt,p,labels=labs,average=None,zero_division=0)
pc=pd.DataFrame({'class':np.array(cls)[labs],'n':[int((gt==l).sum()) for l in labs],
 'Length':f(pl).round(3),'Structure':f(ps).round(3),
 'T':np.mean([f(P[('MTL_Transformer',s)]['test_gene_type'].argmax(1)) for s in seeds],0).round(3),
 'PE':np.mean([f(P[('MTL_Transformer_PE',s)]['test_gene_type'].argmax(1)) for s in seeds],0).round(3),
 'Combined':np.mean([f(p) for p in cpred],0).round(3)}); print(pc.to_string(index=False)); pc.to_csv('/mnt/user-data/outputs/analysis/h60_perclass.csv',index=False)
mx=np.abs(L['maxid']); b=pd.cut(mx,[-0.01,1e-4,0.5,0.65,0.8],labels=['none','<=0.50','0.50-0.65','0.65-0.80'])
print(pd.DataFrame({k[0]:pd.Series(v).groupby(b.codes).mean().round(3).values for k,v in corr.items() if k[1]=='gt'},index=[c for c in b.categories if (b==c).sum()>0]).T, pd.Series(b).value_counts().to_dict())
mf=lambda m: np.mean([f1_score(gt,P[(m,s)]['test_gene_type'].argmax(1),labels=labs,average='macro') for s in seeds])
print('macroF1 PE %.3f T %.3f STL %.3f LEN %.3f STR %.3f COMB %.3f'%(mf('MTL_Transformer_PE'),mf('MTL_Transformer'),mf('STL_GeneType'),f1_score(gt,pl,labels=labs,average='macro'),f1_score(gt,ps,labels=labs,average='macro'),np.mean([f1_score(gt,p,labels=labs,average='macro') for p in cpred])))
pickle.dump({'corr':corr,'mpra':mpra},open('/mnt/user-data/outputs/analysis/h60_corr.pkl','wb'))
