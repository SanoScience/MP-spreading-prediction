import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu

############### regional errors
regional_errors = {}
regional_errors['NDM'] = np.loadtxt('NDM_ALL_regions_22-07-18_21:09:35.csv', delimiter=',')
regional_errors['GCN'] = np.loadtxt('GCN_ALL_regions_22-07-02_20:35:27.csv', delimiter=',')
regional_errors['MAR'] = np.loadtxt('MAR_ALL_regions_22-06-30_06:43:42.csv', delimiter=',')
regional_errors['ESM'] = np.loadtxt('ESM_ALL_regions_22-07-19_05:42:50.csv', delimiter=',')
errors = pd.DataFrame(regional_errors)

markers = {'NDM': 'o', 'GCN': '^', 'MAR': 's', 'ESM': 'D'}

plt.figure(figsize=(20, 10))
for r in regional_errors.keys():
    plt.plot(errors[r], '-', marker=markers[r], markersize=4, label=r, linewidth=1.5, alpha = 1)
#sns.lineplot(data=errors, markers=False, dashes=False, lw=2)
#plt.plot(errors)

plt.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=22)
plt.ylabel('Regional Error', fontsize=18)
plt.yticks(fontsize=14)
plt.xlim(-1, len(regional_errors['MAR']))
plt.xticks(np.arange(len(regional_errors['MAR']), step=2), fontsize=12, rotation=40)
plt.xlabel('ROI (166 AAL3 regions)', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig('All_regional_errors.png')


############### mannwhitneyu tests
mwus_statistics = {}
mwus_pvalues = {}

labels = []
statistics = []
pvalues = []
test = mannwhitneyu(errors['MAR'], errors['NDM'], alternative='less', use_continuity=False) 
labels.append('MAR > NDM')
statistics.append(test.statistic)
pvalues.append(test.pvalue)

test = mannwhitneyu(errors['MAR'], errors['ESM'], alternative='less', use_continuity=False) 
labels.append('MAR > ESM')
statistics.append(test.statistic)
pvalues.append(test.pvalue)

test = mannwhitneyu(errors['MAR'], errors['GCN'], alternative='less', use_continuity=False) 
labels.append('MAR > GCN')
statistics.append(test.statistic)
pvalues.append(test.pvalue)

print(statistics)
print(pvalues)

'''
fig = plt.figure(figsize=(25, 7))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
width = 0.02  # the width of the bars
x = np.arange(len(labels))/10 # the label locations
for p in range(len(pvalues)):
    ax.barh(x[p], pvalues[p], width, label=labels[p])
    #plt.plot(mwus_pvalues[s], label=s+' pvalue')
plt.axvline(x = 0.05, color = 'r', linestyle='--', label = 'significance threshold')
ax.invert_yaxis()
ax.set_yticks(x)
ax.set_yticklabels(labels, rotation=0, fontsize=18)
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize=22)
plt.xticks(np.arange(0.055, step=0.005), fontsize=18)
plt.ylim(-width, x[-1]+width)
plt.xlabel('P-value', fontsize=22)
plt.ylabel('Mann-Whitney U test', fontsize=22)
plt.grid(True)
plt.savefig('Mannwhitneyu_tests.png')
'''

############### regional errors densities
plt.figure(figsize=(20, 10))

sns.set(font_scale=1.6)
sns.set_style("whitegrid")

ax = sns.kdeplot(data=errors, shade=True)
ax.set(xticks=np.arange(0.1, step=0.005), xlim=(0.02, 0.1))
ax.set(xlabel = 'Error')
text = "Mann-Whitney U test\n--------------------\nHypothesis  P-value\n"
for i in range(len(pvalues)):
    text += labels[i] + '   ' + "{:.2e}".format(pvalues[i])
    if i < len(pvalues) -1 : text+= '\n'
t = plt.text(.083,7.7, text, family='monospace', fontsize=22)
t.set_bbox(dict(boxstyle='Round', facecolor='#d2dbeb', alpha=1))
plt.tight_layout()
plt.grid(True)
plt.savefig('Density_regional_errors.png')