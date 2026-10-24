import numpy as np, pandas as pd, pickle, re, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
C={'PE':'#1b6ca8','T':'#5aa0d8','LSTM':'#7f7f7f','CNN':'#bdbdbd','STL':'#9e7bb5','LEN':'#d9822b','STR':'#c0392b','STK':'#2e8b57','NN':'#8c6d31','LR':'#c7a76c'}
# ---------- Fig 1: leakage audit ----------
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1,1.6]})
vals=[7140,7323-7140,1003]; labs=['Identical record\nin training','Exact sequence\nin training','Unseen']
ax[0].bar([0],[vals[0]],color='#c0392b'); ax[0].bar([0],[vals[1]],bottom=[vals[0]],color='#e67e22'); ax[0].bar([0],[vals[2]],bottom=[vals[0]+vals[1]],color='#2e8b57')
ax[0].text(0.45,vals[0]/2,f'Identical record\nin training: {vals[0]:,}',va='center',fontsize=8)
ax[0].text(0.45,vals[0]+vals[1]+vals[2]+250,f'Unseen: {vals[2]:,}',va='bottom',fontsize=8,color='#2e8b57')
ax[0].text(0.45,vals[0]-650,f'Exact sequence\nin training: {vals[1]:,}',va='top',fontsize=8,color='#d35400')
ax[0].set_ylim(0,9400)
ax[0].set_xlim(-0.5,2.4); ax[0].set_xticks([]); ax[0].set_ylabel('Test sequences (original split)'); ax[0].set_title('a  Original test set (n = 8,326)',loc='left',fontsize=9)
m=['Transformer\n(original)','BiLSTM','CNN','LogReg\n6-mer+len','1-NN\nlookup']; full=[.929,.829,.868,.894,.957]; clean=[.811,.796,.749,.753,.650]
x=np.arange(5); ax[1].bar(x-0.2,full,0.4,color='#bdbdbd',label='Full original test'); ax[1].bar(x+0.2,clean,0.4,color='#1b6ca8',label='Unseen subset (n = 1,003)')
ax[1].set_xticks(x); ax[1].set_xticklabels(m,fontsize=8); ax[1].set_ylim(0.5,1); ax[1].set_ylabel('Gene Type accuracy'); ax[1].legend(frameon=False,fontsize=8,loc='upper left',ncol=2)
ax[1].set_title('b  Frozen models on unseen sequences',loc='left',fontsize=9)
plt.tight_layout(); plt.savefig('/tmp/figs/Figure1.png'); plt.close()
# ---------- Fig 2: homology-aware benchmark ----------
A=pd.read_csv('/mnt/user-data/outputs/analysis/abs_acc.csv').set_index('model')
order=[('MTL_Transformer_PE','MTL Transformer\n+PE (order-aware)','PE'),('MTL_Transformer','MTL Transformer\n(order-blind)','T'),('MTL_BiLSTM','MTL BiLSTM','LSTM'),('STL','Single-task\nTransformers','STL'),('MTL_CNN','MTL CNN','CNN'),('Length-only GBM','Length only','LEN'),('LogReg 6-mer+length','LogReg\n6-mer+length','LR'),('1-NN 6-mer','1-NN lookup','NN')]
fig,axs=plt.subplots(1,3,figsize=(7.4,3.3),sharey=True)
for a,(t,tt) in zip(axs,[('gt','Gene Type'),('sf','Symbol Family'),('fr','Functional Roles (exact)')]):
    for i,(k,l,c) in enumerate(order):
        s=A.loc[k,f'{t}_all']; v,lo,hi=[float(z) for z in re.findall(r'[\d.]+',s)]
        a.barh(i,v,color=C[c]); a.text(v+0.01,i,f'{v:.3f}',va='center',fontsize=7)
    a.set_title(tt,fontsize=9); a.set_xlim(0.3,0.95); a.invert_yaxis() if a is axs[0] else None
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([o[1] for o in order],fontsize=7.5)
for a in axs: a.set_xlabel('Accuracy',fontsize=8)
plt.tight_layout(); plt.savefig('/tmp/figs/Figure2.png'); plt.close()
# ---------- Fig 3: identity bins ----------
bins=['None\n(n=2,659)','≤0.50\n(n=638)','0.50–0.65\n(n=413)','0.65–0.80\n(n=468)']
D={'MTL Transformer+PE':([.751,.795,.839,.867],'PE','-'),'MTL Transformer (order-blind)':([.709,.771,.794,.818],'T','-'),'MTL BiLSTM':([.677,.751,.827,.832],'LSTM','-'),'Single-task Transformer':([.661,.742,.751,.773],'STL','-'),'Length only':([.675,.784,.840,.900],'LEN','--'),'1-NN lookup':([.427,.658,.864,.900],'NN','--')}
fig,ax=plt.subplots(figsize=(5.2,3.3))
for k,(v,c,ls) in D.items(): ax.plot(range(4),v,ls,marker='o',color=C[c],label=k,lw=1.6,ms=4)
ax.set_xticks(range(4)); ax.set_xticklabels(bins,fontsize=8); ax.set_xlabel('Identity × coverage to nearest training sequence'); ax.set_ylabel('Gene Type accuracy'); ax.legend(frameon=False,fontsize=7.5); ax.set_ylim(0.4,0.95)
plt.tight_layout(); plt.savefig('/tmp/figs/Figure3.png'); plt.close()
# ---------- Fig 4: per-class F1 ----------
L=np.load('split_labels_test.npz',allow_pickle=True); gt=L['gt']; labs=np.unique(gt); cls=L['gt_classes'][labs]
P=pickle.load(open('/tmp/P.pkl','rb')); Cl=pickle.load(open('/tmp/classic_preds.pkl','rb'))
S=pickle.load(open('/mnt/user-data/outputs/analysis/struct_preds.pkl','rb'))['preds']; K=pickle.load(open('/mnt/user-data/outputs/analysis/stack_preds.pkl','rb'))['stack_pred']
f=lambda p: f1_score(gt,p,labels=labs,average=None,zero_division=0)
seeds=[1,7,21,42,100]
rows={'Length only':(f(Cl['Length-only GBM']['gt']),'LEN'),'Length+GC+structure':(f(S['Length+GC+Structure']),'STR'),
'MTL Transformer (order-blind)':(np.mean([f(P[('MTL_Transformer',s,'gt')]) for s in seeds],0),'T'),'MTL Transformer+PE':(np.mean([f(P[('MTL_Transformer_PE',s,'gt')]) for s in seeds],0),'PE'),
'Transformer+PE + biophysical':(np.mean([f(p) for p in K],0),'STK')}
n=[int((gt==l).sum()) for l in labs]; ordr=np.argsort(n)[::-1]
fig,ax=plt.subplots(figsize=(7.2,3.2)); w=0.16; x=np.arange(len(labs))
for i,(k,(v,c)) in enumerate(rows.items()): ax.bar(x+(i-2)*w,v[ordr],w,color=C[c],label=k)
ax.set_xticks(x); ax.set_xticklabels([f"{cls[j].replace('_','_\n')}\n(n={n[j]})" for j in ordr],fontsize=7); ax.set_ylabel('F1 (Gene Type)'); ax.legend(frameon=False,fontsize=7.5,ncol=3,loc='upper right'); ax.set_ylim(0,1.12)
plt.tight_layout(); plt.savefig('/tmp/figs/Figure4.png'); plt.close()
