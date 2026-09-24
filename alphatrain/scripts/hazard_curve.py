"""Death hazard by turn bucket from eval CSVs (seed,score,turns,capped): h = deaths in bucket / alive at bucket start."""
import csv, sys, numpy as np
edges=[0,100,200,300,400,600,800,1000,1500,2000,3000,5000,10**9]
print(f'{"csv":58s} ' + ' '.join(f'{a}-{b if b<10**9 else "inf"}'.rjust(9) for a,b in zip(edges[:-1],edges[1:])))
for f in sys.argv[1:]:
    t=np.array([int(r['turns']) for r in csv.DictReader(open(f)) if r['capped']=='0'])
    row=[]
    for a,b in zip(edges[:-1],edges[1:]):
        alive=(t>=a).sum(); d=((t>=a)&(t<b)).sum(); row.append(f'{100*d/alive:8.1f}%' if alive>30 else '      -  ')
    print(f'{f.split("/")[-1][:58]:58s} ' + ' '.join(row))
