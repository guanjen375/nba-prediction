import sqlite3
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

current_time = str(time.time())

tensorboard = TensorBoard(log_dir='../../Logs/{}'.format(current_time))
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('../../Models/Trained-Model-OU', save_best_only=True, monitor='val_loss', mode='min')

dataset = "dataset_2012-23"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

OU = data['OU-Cover']
total = data['OU']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU','TEAM_NAME','TEAM_NAME.1','TEAM_ID','TEAM_ID.1'],
          axis=1, inplace=True)
data.drop(['GP_RANK','W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK','FG_PCT_RANK', 
           'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK','FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 
           'DREB_RANK', 'REB_RANK','AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 
           'PF_RANK','PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK'],axis=1,inplace=True)
data.drop(['GP_RANK.1', 'W_RANK.1', 'L_RANK.1','W_PCT_RANK.1', 'MIN_RANK.1', 'FGM_RANK.1', 'FGA_RANK.1',
           'FG_PCT_RANK.1', 'FG3M_RANK.1', 'FG3A_RANK.1', 'FG3_PCT_RANK.1',
           'FTM_RANK.1', 'FTA_RANK.1', 'FT_PCT_RANK.1', 'OREB_RANK.1',
           'DREB_RANK.1', 'REB_RANK.1', 'AST_RANK.1', 'TOV_RANK.1', 'STL_RANK.1',
           'BLK_RANK.1', 'BLKA_RANK.1', 'PF_RANK.1', 'PFD_RANK.1', 'PTS_RANK.1','PLUS_MINUS_RANK.1'],axis=1,inplace=True)

data['OU'] = np.asarray(total)
data = data.values
data = data.astype(float)

x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(OU)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu6))
# model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=32,
          callbacks=[tensorboard, earlyStopping, mcp_save])

print('Done')
