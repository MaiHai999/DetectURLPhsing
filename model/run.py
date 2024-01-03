

from tensorflow.keras.models import load_model
import numpy as np
from model.Sequence2Index import Sequence2Index

model = load_model("/Users/maihai/PycharmProjects/DetectURLPhising/model/save_model.hdf5")
input_sequence = 'http://www.sinduscongoias.com.br/index.php/institucional/missao-visao-valores-negocio-e-politica-da-qualidade'
a = Sequence2Index()
indices = a.sequence_to_index(input_sequence , 200).reshape((1, 200))
predictions = model.predict(indices)
max_index = np.argmax(predictions)
print(a.lable[max_index] , end = "\n\n\n")
