# NOTE: post hoc analysis script as run in the author's environment. Input paths (e.g. /tmp/...) are
# absolute; edit them to point to your copies of the files in data/ and results/ before running.
import numpy as np, pandas as pd, scipy.sparse as sp, itertools, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize, LabelEncoder, MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score,f1_score
df=pd.read_pickle('/tmp/all.pkl').drop_duplicates('NCBIGeneID').drop(columns=['split']).set_index('NCBIGeneID')
s=pd.read_csv('split_homology80.csv'); L=np.load('split_labels_test.npz',allow_pickle=True)
tr=df.loc[s[s.split=='train'].NCBIGeneID]; te=df.loc[L['gene_ids']]
le=LabelEncoder().fit(tr.GeneType); assert (le.classes_==L['gt_classes']).all()
fe=LabelEncoder().fit(tr.fam); mlb=MultiLabelBinarizer().fit(tr.roles.str.split('|'))
assert list(mlb.classes_)==list(L['fr_classes'])
gt,sf,fr=L['gt'],L['sf'],L['fr'].astype(int); assert (le.transform(te.GeneType)==gt).all()
ytr={'gt':le.transform(tr.GeneType),'sf':fe.transform(tr.fam),'fr':mlb.transform(tr.roles.str.split('|'))}
idx={''.join(p):i for i,p in enumerate(itertools.product('ACGT',repeat=6))}
def F(seqs):
    r,c=[],[]
    for i,q in enumerate(seqs):
        q=q[:517]
        for j in range(len(q)-5): r.append(i);c.append(idx[q[j:j+6]])
    return normalize(sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(len(seqs),4096)))
lenf=lambda x: np.log1p(np.minimum(x.L.values,517))[:,None]
A,B=F(tr.seq),F(te.seq)
out={}
def rec(name,pg,pf,pr): out[name]={'gt':pg,'sf':pf,'fr':pr}
# majority
rec('Majority',np.full(len(te),np.bincount(ytr['gt']).argmax()),np.full(len(te),np.bincount(ytr['sf']).argmax()),
    np.tile(pd.DataFrame(ytr['fr']).value_counts().index[0],(len(te),1)))
# length-only
g=lambda y: HistGradientBoostingClassifier(random_state=0).fit(lenf(tr),y).predict(lenf(te))
rec('Length-only GBM',g(ytr['gt']),g(ytr['sf']),np.stack([g(ytr['fr'][:,j]) for j in range(4)],1))
# 1-NN
nn=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute').fit(A); i=nn.kneighbors(B)[1][:,0]
rec('1-NN 6-mer',ytr['gt'][i],ytr['sf'][i],ytr['fr'][i])
# LogReg
Ztr=sp.hstack([A,sp.csr_matrix(lenf(tr))]).tocsr(); Zte=sp.hstack([B,sp.csr_matrix(lenf(te))]).tocsr()
lr=lambda y: LogisticRegression(C=10,max_iter=3000).fit(Ztr,y).predict(Zte)
rec('LogReg 6-mer+length',lr(ytr['gt']),lr(ytr['sf']),np.stack([lr(ytr['fr'][:,j]) for j in range(4)],1))
pickle.dump(out,open('/tmp/classic_preds.pkl','wb'))
labs=np.unique(gt)
for k,v in out.items():
    print(f"{k:22s} GT {accuracy_score(gt,v['gt']):.4f}/{f1_score(gt,v['gt'],labels=labs,average='macro',zero_division=0):.4f}  SF {accuracy_score(sf,v['sf']):.4f}  FR {(v['fr']==fr).all(1).mean():.4f}")
# label redundancy: GT -> SF, FR via train majority map, on test true GT
trd=pd.DataFrame({'gt':ytr['gt'],'sf':ytr['sf'],'fr':[tuple(r) for r in ytr['fr']]})
msf=trd.groupby('gt').sf.agg(lambda x:x.value_counts().index[0]); mfr=trd.groupby('gt').fr.agg(lambda x:x.value_counts().index[0])
print('Oracle true-GT -> SF map: %.4f ; -> FR map: %.4f'%((pd.Series(gt).map(msf).values==sf).mean(),(np.array([mfr[g] for g in gt])==fr).all(1).mean()))
