import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df_bookings = pd.read_csv('bookings.csv')
df_bookings_data = pd.read_csv('bookings_data.csv')
df_customer_data = pd.read_csv('customer_data.csv')
df_hotels_dats = pd.read_csv('hotels_data.csv')
df_payments_data = pd.read_csv('payments_data.csv')
df_train_data = pd.read_csv('train_data.csv')
df_predict_data = pd.read_csv('sample_submission_5.csv')

data = df_customer_data.merge(df_bookings, on = 'customer_id')\
                       .merge(df_bookings_data, on = 'booking_id')\
                       .merge(df_payments_data, on = 'booking_id')\
                       .merge(df_train_data, on = 'booking_id')\
                       .merge(df_hotels_dats, on ='hotel_id')

data_predict = df_customer_data.merge(df_bookings, on = 'customer_id')\
                       .merge(df_bookings_data, on = 'booking_id')\
                       .merge(df_payments_data, on = 'booking_id')\
                       .merge(df_predict_data, on = 'booking_id')\
                       .merge(df_hotels_dats, on ='hotel_id')

for feat in ['booking_create_timestamp', 'booking_approved_at']:
    data[feat] = pd.to_datetime(data[feat], errors = 'raise', utc = False)
    data_predict[feat] = pd.to_datetime(data_predict[feat], errors = 'raise', utc = False)

data.dropna(axis = 0, inplace = True)
data_predict['hotel_photos_qty'].fillna(value= data_predict['hotel_photos_qty'].mean(), inplace=True)
data_predict['hotel_description_length'].fillna(value= data_predict['hotel_description_length'].mean(), inplace=True)
data_predict['hotel_name_length'].fillna(value= data_predict['hotel_name_length'].mean(), inplace=True)
data_predict['hotel_category'].fillna(value= data_predict['hotel_category'].mean(), inplace=True)
data_predict['booking_approved_at'].fillna(value= data_predict['booking_approved_at'].mean(), inplace=True)

i =  0
for contry in ['Japan', 'Russia', 'Slovakia', 'Spain', 'Vietnam', 'UK', 'USA',
       'Cambodia', 'Portugal']:
      data.country[data.country == contry] = i
      data_predict.country[data_predict.country == contry] = i
      i += 1
data.payment_type[data.payment_type == 'credit_card'] = 1
data.payment_type[data.payment_type == 'gift_card'] = 2
data.payment_type[data.payment_type == 'debit_card'] = 3
data.payment_type[data.payment_type == 'voucher'] = 4
data.payment_type[data.payment_type == 'not_defined'] = -1
data_predict.payment_type[data_predict.payment_type == 'credit_card'] = 1
data_predict.payment_type[data_predict.payment_type == 'gift_card'] = 2
data_predict.payment_type[data_predict.payment_type == 'debit_card'] = 3
data_predict.payment_type[data_predict.payment_type == 'voucher'] = 4
data_predict.payment_type[data_predict.payment_type == 'not_defined'] = -1

data.booking_status[data.booking_status == 'completed'] = 1
data.booking_status[data.booking_status == 'canceled'] = -1
data_predict.booking_status[data_predict.booking_status == 'completed'] = 1
data_predict.booking_status[data_predict.booking_status != 1] = -1

data = data.drop_duplicates(['booking_id'], keep= 'last')
data_predict = data_predict.drop_duplicates(['booking_id'], keep= 'last')

timediff = data.booking_approved_at - data.booking_create_timestamp
timediff_predict = data_predict.booking_approved_at - data_predict.booking_create_timestamp

timediff_sec = np.ones(len(timediff))
i = 0
for time in timediff:
    timediff_sec[i] = time/np.timedelta64(1, 's') if time/np.timedelta64(1, 's') > 0 else time/np.timedelta64(1, 's') + 365*84400
    i += 1
timediff_sec_predict = np.ones(len(timediff_predict))
i = 0
for time in timediff_predict:
    timediff_sec_predict[i] = time/np.timedelta64(1, 's') if time/np.timedelta64(1, 's') > 0 else time/np.timedelta64(1, 's') + 365*84400
    i += 1
X = np.array([data.country, data.booking_status, timediff_sec, data.booking_sequence_id, data.price, data.agent_fees, data.payment_sequential, data.payment_type, data.payment_installments, data.hotel_category, data.hotel_name_length, data.hotel_description_length, data.hotel_photos_qty])
y = np.array([data.rating_score])
X = X.T
y = y.T

X_predict = np.array([data_predict.country, data_predict.booking_status, timediff_sec_predict, data_predict.booking_sequence_id, data_predict.price, data_predict.agent_fees, data_predict.payment_sequential, data_predict.payment_type, data_predict.payment_installments, data_predict.hotel_category, data_predict.hotel_name_length, data_predict.hotel_description_length, data_predict.hotel_photos_qty])
X_predict = X_predict.T

clf = LinearRegression()
clf.fit(X, y)
y_pred_lin = clf.predict(X_predict)

booking_id_predict = data_predict.booking_id

y_pred_lin = (np.round(y_pred_lin))
y_pred_lin = y_pred_lin.ravel()
d = {'booking_id' : booking_id_predict, 'rating_score' : y_pred_lin}
df = pd.DataFrame(data = d)
df_final = df_predict_data.merge(df, on = 'booking_id', how = 'left')
df_final = df_final.drop(columns = ['rating_score_x'])
df_final = df_final.rename(columns = {'rating_score_y' : 'rating_score'})
df_final['rating_score'].fillna(value  = 4, inplace = True)
df_final.to_csv('ce20b085_be18b014.csv', index = False)
