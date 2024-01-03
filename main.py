from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from tensorflow.keras.models import load_model
import numpy as np
from model.Sequence2Index import Sequence2Index
from flask import jsonify


model = load_model("/Users/maihai/PycharmProjects/DetectURLPhising/model/save_model.hdf5")
input2index = Sequence2Index()

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/detect', methods=['GET'] )
@cross_origin(origin='*')
def detectURL():
    url = request.args.get('URL')
    indices = input2index.sequence_to_index(url , 200).reshape((1, 200))
    predictions = model.predict(indices)
    max_index = np.argmax(predictions)
    result = input2index.lable[max_index]
    return result


# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9999')

