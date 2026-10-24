import numpy as np, pandas as pd, pickle
from sklearn.metrics import f1_score
P=pickle.load(open('/tmp/P.pkl','rb')); C=pickle.load(open('/tmp/classic_preds.pkl','rb'))
L=np.load('split_labels_test.npz',allow_pickle=True); gt,sf,fr=L['gt'],L['sf'],L['fr'].astype(int)
s=pd.read_csv('split_homology80.csv').set_index('NCBIGeneID'); clu=s.loc[L['gene_ids'],'cluster'].values
df=pd.read_pickle('/tmp/all.pkl').drop_duplicates('NCBIGeneID').set_index('NCBIGeneID').loc[L['gene_ids']]
mpra=(df.Description.str.contains('Sharpr-MPRA')&(df.L==295)).values
print('Sharpr-MPRA 295-nt tiles in test:',mpra.sum(),'of',len(gt))
seeds=[1,7,21,42,100]
def corr(m,t):   # per-sample correctness averaged over seeds (or deterministic classic)
    y={'gt':gt,'sf':sf,'fr':fr}[t]
    if m in C: p=C[m][t]; return ((p==y).all(1) if t=='fr' else (p==y)).astype(float)
    return np.mean([((P[(m,sd,t)]==y).all(1) if t=='fr' else (P[(m,sd,t)]==y)) for sd in seeds],0)
uc=np.unique(clu); members=[np.where(clu==c)[0] for c in uc]
rng=np.random.default_rng(0)
B=[np.concatenate([members[j] for j in rng.integers(0,len(uc),len(uc))]) for _ in range(2000)]
def ci(a,b=None,mask=None):
    m=np.ones(len(gt),bool) if mask is None else mask
    d=a if b is None else a-b
    est=d[m].mean(); bs=[]
    for ix in B:
        ix=ix[m[ix]]; bs.append(d[ix].mean())
    lo,hi=np.percentile(bs,[2.5,97.5]); p=2*min((np.array(bs)<=0).mean(),(np.array(bs)>=0).mean())
    return est,lo,hi,p
M={'T':'MTL_Transformer','PE':'MTL_Transformer_PE','CNN':'MTL_CNN','LSTM':'MTL_BiLSTM','LEN':'Length-only GBM','LR':'LogReg 6-mer+length','NN':'1-NN 6-mer'}
STL={'gt':'STL_GeneType','sf':'STL_SymbolFamily','fr':'STL_FunctionalRoles'}
rows=[]
for t in ['gt','sf','fr']:
    for sub,mask in [('all',None),('non-MPRA',~mpra)]:
        for a,b in [('T','STL'),('PE','T'),('T','LSTM'),('T','CNN'),('T','LEN'),('PE','LEN'),('PE','LSTM')]:
            ca=corr(M[a],t); cb=corr(STL[t] if b=='STL' else M[b],t)
            e,lo,hi,p=ci(ca,cb,mask)
            rows.append({'task':t,'subset':sub,'contrast':f"{a} - {b}",'diff':e,'lo':lo,'hi':hi,'p_boot':p})
R=pd.DataFrame(rows); R.to_csv('/tmp/contrasts.csv',index=False)
pd.set_option('display.width',200); print(R.round(4).to_string(index=False))
# absolute accuracies with CI per model, all and non-MPRA
rows=[]
for name in ['MTL_Transformer_PE','MTL_Transformer','MTL_BiLSTM','MTL_CNN','STL','Length-only GBM','LogReg 6-mer+length','1-NN 6-mer','Majority']:
    r={'model':name}
    for t in ['gt','sf','fr']:
        nm=STL[t] if name=='STL' else name
        c=corr(nm,t)
        for sub,mask in [('all',None),('nonMPRA',~mpra)]:
            e,lo,hi,_=ci(c,None,mask); r[f'{t}_{sub}']=f"{e:.3f} [{lo:.3f},{hi:.3f}]"
    rows.append(r)
A=pd.DataFrame(rows); A.to_csv('/tmp/abs_acc.csv',index=False); print(A.to_string(index=False))
np.save('/tmp/mpra.npy',mpra)
