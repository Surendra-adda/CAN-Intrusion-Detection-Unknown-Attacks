from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / 'images'
OUT.mkdir(exist_ok=True)

# Fig. 1: architecture diagram
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis('off')
blocks = [
    (0.2, 1.08, 1.65, 1.0, 'CAN captures', 'ID, DLC, payload, time'),
    (2.2, 1.08, 1.75, 1.0, 'Feature pipeline', '13 numeric features'),
    (4.3, 1.08, 1.75, 1.0, 'Known-class MLP', '13 → 128 → 64 → 6'),
    (6.4, 1.08, 1.75, 1.0, 'Hidden embedding', '64-D representation'),
    (8.5, 1.08, 1.55, 1.0, 'Benign-only IF', '300 trees'),
    (10.4, 1.08, 1.4, 1.0, 'Decision', 'score > τ?'),
]
for x,y,w,h,title,sub in blocks:
    patch = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.08',linewidth=1.3,facecolor='#eef3f8',edgecolor='#345')
    ax.add_patch(patch)
    ax.text(x+w/2,y+0.63,title,ha='center',va='center',fontsize=10,fontweight='bold')
    ax.text(x+w/2,y+0.30,sub,ha='center',va='center',fontsize=8.5)
for i in range(len(blocks)-1):
    x1=blocks[i][0]+blocks[i][2]+0.03; x2=blocks[i+1][0]-0.04
    ax.add_patch(FancyArrowPatch((x1,1.58),(x2,1.58),arrowstyle='-|>',mutation_scale=14,linewidth=1.25,color='#345'))
ax.text(5.15,2.55,'Supervised learning on six known classes',ha='center',fontsize=9,color='#334')
ax.text(9.25,2.55,'Unsupervised fit on benign embeddings only',ha='center',fontsize=9,color='#334')
ax.text(6.0,0.42,'Calibration: 99th percentile of benign calibration scores; 97.5th percentile retained as exploratory analysis',ha='center',fontsize=8.7)
fig.tight_layout(); fig.savefig(OUT/'pipeline.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Fig. 2: family recall comparison
families=['DOUBLE','FUZZING','INTERVAL','SPEED','SYSTEMATIC','TRIPLE']
r99=np.array([0,65.27,0,0,5.37,0])
r975=np.array([9.06,80.52,96.42,60.27,34.32,9.69])
x=np.arange(len(families)); w=0.36
fig,ax=plt.subplots(figsize=(9,4.4))
ax.bar(x-w/2,r99,w,label='99th percentile (primary)')
ax.bar(x+w/2,r975,w,label='97.5th percentile (exploratory)')
ax.set_ylabel('Recall (%)'); ax.set_ylim(0,110); ax.set_xticks(x, families)
ax.grid(axis='y',alpha=.25); ax.legend(frameon=False,loc='upper left')
ax.set_title('Unknown-attack recall varies strongly by family')
fig.tight_layout(); fig.savefig(OUT/'family_recall.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Fig. 3: confusion matrices two panels
mats=[np.array([[852507,8937],[16869,3956]]),np.array([[842539,18905],[7803,13022]])]
titles=['Primary: 99th percentile','Exploratory: 97.5th percentile']
fig,axes=plt.subplots(1,2,figsize=(9,4.0))
for ax,mat,title in zip(axes,mats,titles):
    ax.imshow(mat,cmap='Blues')
    ax.set_title(title,fontsize=10)
    ax.set_xticks([0,1],['Pred. benign','Pred. attack'],rotation=15)
    ax.set_yticks([0,1],['Actual benign','Actual attack'])
    for (i,j),v in np.ndenumerate(mat):
        ax.text(j,i,f'{v:,}',ha='center',va='center',fontsize=9,color='black')
fig.suptitle('Confusion matrices on the same processed unknown-test file',fontsize=11)
fig.tight_layout(); fig.savefig(OUT/'confusion_matrices.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Fig. 4: score medians
labels=['BENIGN','DOUBLE','FUZZING','INTERVAL','SPEED','SYSTEMATIC','TRIPLE']
medians=np.array([-0.131440,-0.108853,0.297320,0.044970,0.033780,-0.029677,0.006674])
fig,ax=plt.subplots(figsize=(8.8,4.2))
ax.bar(labels,medians)
ax.axhline(0,color='black',linewidth=.8)
ax.set_ylabel('Median anomaly score')
ax.set_title('Observed median anomaly score by traffic label')
ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(OUT/'score_medians.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Fig. 5: main metrics by model / operating point
methods=['Raw-feature IF\n99th','Embedding IF\n99th','Embedding IF\n97.5th*']
f1=[28.94,23.47,49.37]; recall=[24.47,19.00,62.53]; fpr=[1.08,1.04,2.19]
x=np.arange(3)
fig,axes=plt.subplots(1,2,figsize=(9.4,4.0))
axes[0].bar(x-.18,f1,.36,label='F1 (%)')
axes[0].bar(x+.18,recall,.36,label='Recall (%)')
axes[0].set_xticks(x,methods); axes[0].set_ylabel('Percent'); axes[0].set_ylim(0,75); axes[0].legend(frameon=False); axes[0].grid(axis='y',alpha=.2)
axes[0].set_title('Detection effectiveness')
axes[1].bar(x,fpr,width=.55)
axes[1].set_xticks(x,methods); axes[1].set_ylabel('False-positive rate (%)'); axes[1].set_ylim(0,3); axes[1].grid(axis='y',alpha=.2)
axes[1].set_title('False alarms')
fig.suptitle('*97.5th percentile is exploratory, not confirmatory')
fig.tight_layout(); fig.savefig(OUT/'method_comparison.png',dpi=220,bbox_inches='tight'); plt.close(fig)
