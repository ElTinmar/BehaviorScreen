import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sheet_id = "1UPuJvXpHCQQAw8_jbeM0mtYGDC1RDPkqAYdKLDY60Ls"
tab_gid = "1999968906"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={tab_gid}"
tested = pd.read_csv(url, usecols=range(5), skiprows=1)
lethal = pd.read_csv(url, usecols=[7,8,9], skiprows=1)

tested = tested.rename(columns={'N control': 'Control', 'N ronidazole': 'Ablated'})
df_melted = tested.melt(id_vars=['line'], value_vars=['Control', 'Ablated'], 
                    var_name='Group', value_name='Sample Size (N)')

total_control = int(tested['Control'].sum(skipna=True))
total_ablated = int(tested['Ablated'].sum(skipna=True))
total_fish = total_control + total_ablated 
fish_minutes = fish_minutes = total_fish * 4490 // 60
total_hours, minutes = divmod(fish_minutes, 60)
days, hours = divmod(total_hours, 24)

plt.figure(figsize=(14, 6))
sns.barplot(data=df_melted, x='line', y='Sample Size (N)', hue='Group', 
            palette=['#1f77b4', '#d62728'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Line', fontsize=12)
plt.ylabel('N', fontsize=12)
text_str = f"Control N = {total_control}\nAblated N = {total_ablated}\n{days} days, {hours} hours, and {minutes} minutes"
plt.gca().text(0.02, 0.95, text_str, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top')
plt.tight_layout()
plt.show()
plt.savefig('line_sample_sizes.png', dpi=300)