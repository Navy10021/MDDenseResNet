import pandas as pd 
import os 
from models.hex import *

def convert_image_from_df(df, base_path):

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for idx, row in df.iterrows():
        # get data and label
        hex_data = row['16_bytes']
        class_label = row['Class']
        img_id = row['Id']

        # convert hex data to image
        img = hex_to_image(hex_data, width=1024)

        # Create directory for the class if it doesn't exist
        class_directory = os.path.join(base_path, str(class_label))
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)

        # Save the image
        img.save(os.path.join(class_directory, f'{img_id}.png'))
        

#path = './data/under_sample_400.csv'
path = './data/over_sample_max.csv'
oversampled_df = pd.read_csv(path)
oversampled_df.tail()
print("16_bytes's width * height : ", np.sqrt(oversampled_df['Size'].mean()))

# Convert 16_bytes to image
convert_image_from_df(oversampled_df, './data/img/sample')
print("The image data conversion of the Oversampled dataset has been completed.")

#convert_image_from_df(train_df, './data/img/train')
#print("The image data conversion of the Oversampled Train dataset has been completed.")

#convert_image_from_df(test_df, './data/img/test')
#print("The image data conversion of the Oversampled Test dataset has been completed.")