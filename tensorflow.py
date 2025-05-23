from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorial
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.callbacks import TensorBoard

sequences, labels =[],[]

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
            window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

log_dir = os.path.join('logs')
tb_callback = TensorBoard(log_dir = log_dir)
model = Sequential
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#tensorboard --logdir=.
model.compile(optimizers='Adam',loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train,y_train, epochs=2000, callbacks =[tb_callback])
model.summary
model.predict(X_test)

actions[np.argmax(res[4])]
actions[np.argmax(y_test[4])]

#save weight
model.save('action.h5')
del model
model.load_weights('action.h5')


from sklearn.metrics import multilabel_confusion_matrix,accuracy_score

ywhat