import numpy as np
import matplotlib.pyplot as plt

baseline = np.loadtxt('sub-AD4009_ses-baseline_acq-AP_date-2011-07-07_trc-av45_pet.csv', delimiter=',')
followup = np.loadtxt('sub-AD4009_ses-followup_acq-AP_date-2013-07-03_trc-av45_pet.csv', delimiter=',')

prediction = followup + np.random.normal(0, .025, size=followup.shape)
plt.figure(figsize=(20,10))
#plt.plot(baseline, '-', marker='o', c='#390099', label='baseline', linewidth=1)
plt.plot(followup, '-', marker='o', c='#00A6FB', linewidth=1)
plt.plot(prediction, '-', marker='o', c='#FF0054', linewidth=6, alpha = 0.2)
#plt.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=22)
#plt.ylabel('Regional Error', fontsize=18)
#plt.yticks(fontsize=14)
plt.xlim(-1, len(baseline))
plt.tick_params(
    axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False) 
#plt.xticks(np.arange(len(baseline), step=2), fontsize=12, rotation=40)
#plt.xlabel('ROI (166 AAL3 regions)', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig('followup.png')
#plt.savefig('baseline.png')