import pandas as pd
import sys
import os 
sys.path.append(os.getcwd())

csv_file1 = 'single_task_results.csv'
csv_file2 = 'results_quant.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

df_combined = pd.concat([df1, df2], axis=1)

output_file = 'combined_file.csv'
df_combined.to_csv(output_file, index=False)

print(f"Combined file saved as {output_file}")
