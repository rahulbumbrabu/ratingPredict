import pandas as pd
import matplotlib.pyplot as plt
df_train_data = pd.read_csv('train_data.csv')
df_payments_data = pd.read_csv('payments_data.csv')
df_train_data = pd.merge(df_train_data, df_payments_data, on = 'booking_id' , how = 'inner')
