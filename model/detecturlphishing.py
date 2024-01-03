import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D

RATIO = 0.7
LENGTH_SENTENCE = 200

path = "/Users/maihai/PycharmProjects/DetectURLPhising/dataset/malicious_phish.csv"
df = pd.read_csv(path, on_bad_lines = "skip")

def preprocess_data(df):
    df.loc[df['type'] == 'phishing', 'type'] = 1
    df.loc[df['type'] == 'defacement', 'type'] = 2
    df.loc[df['type'] == 'malware', 'type'] = 3
    df.dropna(inplace=True)
    df_filtered = df.drop(df[df['type'] == 'benign'].index)
    df_filtered['type'] = df_filtered['type'].astype(int)

    return df_filtered

df_phising =  preprocess_data(df)
arr_phising = df_phising.values

path_phising = "/Users/maihai/PycharmProjects/DetectURLPhising/dataset/HiddenFraudulentURLs.csv"
df_phising1 = pd.read_csv(path_phising, header=0, sep=';', on_bad_lines = "skip")

def preprocess_data_new(df):
   df = df.iloc[:, [0, 1]].copy()
   df.loc[df['compromissionType'] == 'phishing', 'compromissionType'] = 1
   df.loc[df['compromissionType'] == 'defacement', 'compromissionType'] = 2
   df.loc[df['compromissionType'] == 'normal', 'compromissionType'] = 0
   df['compromissionType'] = df['compromissionType'].astype(int)
   return df

df_phising_new = preprocess_data_new(df_phising1)
arr_phising_new = df_phising_new.values

# Mở file để đọc
file_path = '/Users/maihai/PycharmProjects/DetectURLPhising/dataset/name_and_url.txt'
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

data_url_beginer = []
for line in lines:
  try:
    name, url = line.split("\t")
    data_url_beginer.append([url,0])
  except ValueError:
    continue

data_url_bengin = np.array(data_url_beginer)

def create_character_vocabulary():
    # Define the character vocabulary based on the provided rules
    character_vocabulary = {}
    index_counter = 1  # Start index from 1

    # Lowercase letters
    for char in 'abcdefghijklmnopqrstuvwxyz':
        character_vocabulary[char] = index_counter
        index_counter += 1

    # Uppercase letters
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        character_vocabulary[char] = index_counter
        index_counter += 1

  # Digits
    for char in '0123456789':
        character_vocabulary[char] = index_counter
        index_counter += 1

    # Special characters
    special_characters = ',;.!?:”’/\\|_@#$%^&*~`+-=<>()[]{}:'
    for char in special_characters:
        character_vocabulary[char] = index_counter
        index_counter += 1

    return character_vocabulary

character_vocabulary = create_character_vocabulary()

def char_to_index(character):
    return character_vocabulary.get(character, 95)  # -1 for unknown characters

def sequence_to_index(input_sequence , max_len):
    indices = [char_to_index(char) for char in input_sequence]
    padded_indices = pad_sequences([indices], maxlen=max_len, padding='post', value=0)[0]
    return np.array(padded_indices)

def processing_data(data , length_sentence):
    filtered_list = np.array([item for item in data if len(item[0]) <= length_sentence])
    x = filtered_list[:,0]
    max_length = len(max(x, key=len))
    x1 = [sequence_to_index(sequence, max_length) for sequence in x]
    x2 = np.array(x1)

    y = filtered_list[:,1]
    y2 = to_categorical(y)
    return x2 , y2




input_sequence = 'Hải'
indices = sequence_to_index(input_sequence , 9)
print(f"The indices of '{input_sequence}' are: {indices}")

data = np.concatenate((arr_phising, data_url_bengin ,arr_phising_new ))
shuffle_order = list(range(len(data)))
random.shuffle(shuffle_order)

data = data[shuffle_order]

train_set = data[:int(len(data)*RATIO)]
test_set = data[int(len(data)*RATIO):]


x_train , y_train = processing_data(train_set , LENGTH_SENTENCE)
x_test , y_test = processing_data(test_set , LENGTH_SENTENCE)


model = Sequential()
model.add(Embedding(input_dim=96, output_dim=64, input_length=LENGTH_SENTENCE))
model.add(Conv1D(254, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3 , activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3 , activation='relu'))
model.add(MaxPooling1D(2))
model.add(LSTM(units = 64))
model.add(Dropout(0.5))
model.add(Dense(28 , activation = "relu"))
model.add(Dense(y_train.shape[1] , activation = "sigmoid"))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

save_path = "/Users/maihai/PycharmProjects/DetectURLPhising/model/save_model.hdf5"
best_model = ModelCheckpoint(save_path,monitor='loss',verbose=2,save_best_only=True,mode='auto')
history = model.fit(x_train, y_train, epochs=7, batch_size=32, validation_data=(x_test, y_test) , callbacks=[best_model])

# Plot training loss and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training accuracy and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



