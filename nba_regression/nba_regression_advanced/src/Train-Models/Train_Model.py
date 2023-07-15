import sqlite3
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

current_time = str(time.time())

tensorboard = TensorBoard(log_dir='../../Logs/{}'.format(current_time))
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('../../Models/Trained-Model-ML', save_best_only=True, monitor='val_loss', mode='min')

dataset = "dataset_2012-23"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

scores = data['Score']
margin = data['Home-Team-Win']

data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU','TEAM_ID', 'TEAM_ID.1'],
          axis=1, inplace=True)
data.drop(['TM_TOV_PCT_RANK','EFG_PCT_RANK','TS_PCT_RANK' ,'PACE_RANK','PIE_RANK',
           'AST_TO_RANK','AST_RATIO_RANK','OREB_PCT_RANK','DREB_PCT_RANK','REB_PCT_RANK',
            'MIN_RANK','OFF_RATING_RANK','DEF_RATING_RANK','NET_RATING_RANK','AST_PCT_RANK',
            'GP_RANK','W_RANK','L_RANK','W_PCT_RANK'],axis=1,inplace=True)
data.drop(['TM_TOV_PCT_RANK.1','EFG_PCT_RANK.1','TS_PCT_RANK.1' ,'PACE_RANK.1','PIE_RANK.1',
           'AST_TO_RANK.1','AST_RATIO_RANK.1','OREB_PCT_RANK.1','DREB_PCT_RANK.1','REB_PCT_RANK.1',
           'MIN_RANK.1','OFF_RATING_RANK.1','DEF_RATING_RANK.1','NET_RATING_RANK.1','AST_PCT_RANK.1',
           'GP_RANK.1','W_RANK.1','L_RANK.1','W_PCT_RANK.1'],axis=1,inplace=True)


data = data.values
data = data.astype(float)

x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(margin)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=32,
          callbacks=[tensorboard, earlyStopping, mcp_save])

print('Done')
