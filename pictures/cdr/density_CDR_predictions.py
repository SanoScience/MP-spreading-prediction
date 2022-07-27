import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import pandas as pd

cdrs = pd.read_csv('assigned_true_CDR.csv', delimiter=',')
print(cdrs)
test = mannwhitneyu(cdrs['predicted_CDR'], cdrs['true_CDR'], alternative='greater', use_continuity=False) 
pvalue = test.pvalue
print(pvalue)


############### cdrs density
plt.figure(figsize=(20, 10))

sns.set(font_scale=1.6)
sns.set_style("whitegrid")

ax = sns.kdeplot(data=cdrs, shade=True)
ax.set(xlim=(0, 3.5))
ax.set(xlabel = 'CDR')
plt.tight_layout()
plt.grid(True)
text = "Mann-Whitney U test\n-----------------------------------\nHypothesis                 P-value\nPredicted CDR > True CDR   " + "{:.2e}".format(pvalue)
t = plt.text(2.24,0.625, text, family='monospace', fontsize=22)
t.set_bbox(dict(boxstyle='Round', facecolor='#d2dbeb', alpha=1))
plt.savefig('Density_CDRs.png')

############### mannwhitneyu tests

fig = plt.figure(figsize=(21, 7))
ax = fig.add_axes([0.1, 0.1, 0.4, 0.7]) # main axes
width = 0.001  # the width of the bars
ax.barh(0, pvalue, width, label='Predicted vs True')
    #plt.plot(mwus_pvalues[s], label=s+' pvalue')
#plt.axvline(x = 0.05, color = 'r', linestyle='--', label = 'significance threshold')
ax.invert_yaxis()
ax.set_yticks([0])
ax.set_yticklabels(['Predicted vs\n True CDR'], rotation=0, fontsize=18)
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize=22)
#plt.xticks(np.arange(0.055, step=0.005), fontsize=18)
plt.ylim(-0.01, +0.01)
plt.xlabel('P-value', fontsize=22)
plt.ylabel('Mann-Whitney U test', fontsize=22)
plt.grid(True)
plt.savefig('CDR_Mannwhitneyu_tests.png')