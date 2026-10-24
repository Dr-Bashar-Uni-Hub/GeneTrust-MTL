import RNA, pandas as pd, pickle, os, time
out='/mnt/user-data/outputs/analysis/mfe200_mouse.pkl'
M=pd.read_csv('/tmp/mz/mouse_external.csv')
done=pickle.load(open(out,'rb')) if os.path.exists(out) else {}
t=time.time(); n=0
for g,q in zip(M.NCBIGeneID,M.seq):
    if g in done: continue
    w=q[:200].replace('T','U'); ss,mfe=RNA.fold(w); done[g]=(mfe,len(w),ss.count('(')*2/len(w)); n+=1
    if time.time()-t>280: break
pickle.dump(done,open(out,'wb')); print('folded',n,'total',len(done),'of',len(M))
