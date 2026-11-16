# NOTE: post hoc analysis script as run in the author's environment. Input paths (e.g. /tmp/...) are
# absolute; edit them to point to your copies of the files in data/ and results/ before running.
# Strict mouse subsets defined with the SAME rule as the human partitions:
# a mouse gene is excluded if any alignment to a human training sequence has identity >= thr
# and covers >= 80% of EITHER sequence (thr = 0.8 or 0.6).
# Search (run once; -k 11/--split-memory-limit only needed on low-memory machines):
#   mmseqs easy-search m.fa h.fa hits.m8 tmp --search-type 3 --min-seq-id 0.6 -c 0 --max-seqs 300 -s 7.5 \
#     --format-output "query,target,fident,qstart,qend,tstart,tend,qlen,tlen"
import pandas as pd
h = pd.read_csv('hits.m8', sep='\t', header=None, names=['q','t','fid','qs','qe','ts','te','ql','tl'])
h['qcov'] = ((h.qe - h.qs).abs() + 1) / h.ql
h['tcov'] = ((h.te - h.ts).abs() + 1) / h.tl
def matched(thr):
    x = h[(h.fid >= thr) & ((h.qcov >= 0.8) | (h.tcov >= 0.8))]
    return set(x.q.str[1:].astype(int))
M = pd.read_csv('data/mouse_external.csv')
out = pd.DataFrame({'NCBIGeneID': M.NCBIGeneID,
                    'no_rule80_match': ~M.NCBIGeneID.isin(matched(0.8)),
                    'no_rule60_match': ~M.NCBIGeneID.isin(matched(0.6))})
out.to_csv('results/tables/mouse_strict_rule_based.csv', index=False)
print(out[['no_rule80_match', 'no_rule60_match']].sum())   # expected: 10,102 and 8,377
