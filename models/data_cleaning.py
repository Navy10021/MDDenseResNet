import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from models.hex import *

path = './data/malware.csv'
df = pd.read_csv(path)

# 1. Check data infomation
df.info()
print("Malware mean size : ", df['Size'].mean())
print("16_bytes's width * height : ", np.sqrt(df['Size'].mean()))

# 2. Histogram of data volume
plt.figure(figsize=(10, 6))
sns.histplot(df['Class'], bins=30, kde=True)
plt.title('Histogram of Malware')
plt.xlabel('Class')
plt.ylabel('Volume')
plt.show()

# 3. Box plot of each malware size
plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Size', data=df)
plt.title('Malware Size by Class')
plt.xlabel('Class')
plt.ylabel('Malware Size')
plt.show()

# 4. Oversampling and Undersampling
# Oversampling or Undersampling
class_counts = df['Class'].value_counts()
min_samples = class_counts.median()
#min_samples = class_counts.max()
print("Minimum number of Sampling : ", min_samples)
# Oversampling strategy: Bring up minority classes to the median count
oversampling_strategy = {class_label: int(min_samples) for class_label, count in class_counts.items() if count < min_samples}
# Undersampling strategy: Reduce majority classes to the median count
undersampling_strategy = {class_label: int(min_samples) for class_label, count in class_counts.items() if count > min_samples}

ros = RandomOverSampler(sampling_strategy=oversampling_strategy, random_state=42)
rus = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)

# Proceed with resampling as before
x_over, y_over = ros.fit_resample(df[['Id','16_bytes', 'Size']], df['Class'])
x_under, y_under = rus.fit_resample(df[['Id','16_bytes', 'Size']], df['Class']) if undersampling_strategy else (df[['Id','16_bytes', 'size']], df['Class'])

oversampled_df = pd.DataFrame(x_over, columns=['Id','16_bytes', 'Size'])
oversampled_df['Class'] = y_over

if undersampling_strategy:
    undersampled_df = pd.DataFrame(x_under, columns=['Id','16_bytes', 'Size'])
    undersampled_df['Class'] = y_under
else:
    undersampled_df = pd.DataFrame([], columns=['Id','16_bytes', 'Size', 'Class'])

# Visualization
def plot_class_distribution(df, title):
    # Count the occurrences of each class
    class_counts = df['Class'].value_counts()

    # Create a bar plot
    sns.barplot(x=class_counts.index, y=class_counts.values)

    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

plot_class_distribution(df, 'Original Dataset Class Distribution')
plot_class_distribution(oversampled_df, 'Oversampled Dataset Class Distribution')
plot_class_distribution(undersampled_df, 'Undersampled Dataset Class Distribution')
print("Oversampled data's 16_bytes's width * height : ", np.sqrt(oversampled_df['Size'].mean()))
print("Undersampled data's 16_bytes's width * height : ", np.sqrt(undersampled_df['Size'].mean()))

# 5. Undersampling dataset
undersampled_df.to_csv("./data/under_sample.csv", index=False)

# 6. Splitting Oversampled Train-Test dataset
X = oversampled_df.drop('Class', axis=1)
y = oversampled_df['Class']

# indicates that 10% of the data will be set aside for the test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_df = pd.DataFrame(X_train)
train_df['Class'] = y_train

test_df = pd.DataFrame(X_test)
test_df['Class'] = y_test

print("Training set size:", len(train_df))
print("Testing set size:", len(test_df))

# Save to CSV
train_df.to_csv('./data/train.csv', index=False)
test_df.to_csv('./data/test.csv', index=False)