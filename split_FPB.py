import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'financial_phrase_bank_final.csv'
data = pd.read_csv(file_path)

train_data, validation_data = train_test_split(data, test_size=0.3, random_state=42)


train_file_path = 'FPB_training_set_final.csv'
validation_file_path = 'FPB_validation_set_final.csv'

train_data.to_csv(train_file_path, index=False)
validation_data.to_csv(validation_file_path, index=False)

