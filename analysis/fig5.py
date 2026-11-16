# NOTE: post hoc analysis script as run in the author's environment. Input paths (e.g. /tmp/...) are
# absolute; edit them to point to your copies of the files in data/ and results/ before running.
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
exec(open('/tmp/motif.py').read().split("rng=np.random.default_rng(0); rows=[]")[0])
fig,ax=plt.subplots(1,2,figsize=(7.4,3.1),gridspec_kw={'width_ratios':[1.15,1]})
M=pd.read_csv('/mnt/user-data/outputs/analysis/motif_enrichment.csv')
gn=['Order-aware (PE), 5 seeds','Order-blind (original), 5 seeds','Untrained PE control']; gc=['#1b6ca8','#5aa0d8','#bdbdbd']
lab=['Order-aware Transformer (5 seeds)','Order-blind Transformer (5 seeds)','Untrained control']
mot=['tRNA A box','tRNA B box','snoRNA C box','snoRNA D box']; x=np.arange(4)
for i,(g,c,l) in enumerate(zip(gn,gc,lab)):
    v=[M[(M.motif==mm)&(M.model==g)].median_enrichment.values[0] for mm in mot]; ax[0].bar(x+(i-1)*0.27,v,0.27,color=c,label=l)
ax[0].axhline(0,color='k',lw=0.6); ax[0].set_xticks(x); ax[0].set_xticklabels([m.replace(' ','\n',1) for m in mot],fontsize=8); ax[0].set_ylabel('Median motif enrichment\n(vs. random windows)'); ax[0].legend(frameon=False,fontsize=7)
ax[0].set_title('a  Attribution to canonical motifs',loc='left',fontsize=9)
tr=np.where((cls=='tRNA')&(pos['tRNA A box']>=0)&(pos['tRNA B box']>=0))[0]
def prof(tl):
    out=np.full((len(tr),72),np.nan)
    for r,i in enumerate(tr):
        v=np.mean([NA[t][i] for t in tl],0)*len(seqs[i]); n=min(72,len(v)); out[r,:n]=v[:n]
    return np.nanmean(out,0)
for tl,c,l in [(groups['Order-aware (PE), 5 seeds'],gc[0],lab[0]),(groups['Order-blind (original), 5 seeds'],gc[1],lab[1]),(groups['Untrained PE control'],gc[2],lab[2])]:
    ax[1].plot(np.arange(1,73),prof(tl),color=c,lw=1.4,label=l)
a0=int(np.median(pos['tRNA A box'][tr])); b0=int(np.median(pos['tRNA B box'][tr]))
ax[1].axvspan(a0+1,a0+11,color='#f4d03f',alpha=0.35,lw=0); ax[1].axvspan(b0+1,b0+9,color='#f4d03f',alpha=0.35,lw=0)
ax[1].text(a0+6,ax[1].get_ylim()[1]*0.92,'A box',ha='center',fontsize=8); ax[1].text(b0+5,ax[1].get_ylim()[1]*0.92,'B box',ha='center',fontsize=8)
ax[1].axhline(1,color='k',lw=0.5,ls=':'); ax[1].set_xlabel('Nucleotide position (tRNA)'); ax[1].set_ylabel('Mean relative attribution\n(1 = uniform)')
ax[1].set_title(f'b  tRNA attribution profile (n = {len(tr)})',loc='left',fontsize=9)
plt.tight_layout(); plt.savefig('/tmp/figs/Figure5.png'); plt.close(); print('ok',len(tr),a0,b0)
