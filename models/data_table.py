import pandas as pd 
import os 
from models.hex import *

path = './data/bytes'
files = [f for f in os.listdir(path) if f.endswith('.bytes')]
print("Number of malware data : ", len(files))
data = []
for file_name in files:
    file_path = os.path.join(path, file_name)
    hex_data = read_hex_file(file_path)

    data.append({'Id' : file_name.replace('.bytes', ''), '16_bytes' : hex_data})

data_df = pd.DataFrame(data)
labels_df = pd.read_csv('./data/trainLabels.csv')

# Perform an inner join on the 'Id' column
inner_join_df = pd.merge(data_df, labels_df, on = 'Id', how='inner')
inner_join_df['Size'] = inner_join_df['16_bytes'].apply(len)
inner_join_df.to_csv("./data/malware.csv", index=False)