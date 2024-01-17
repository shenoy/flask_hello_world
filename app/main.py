from flask import Flask
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import os
from PIL import Image



app = Flask(__name__)

def hello_world():
    return 'Hello, World!'
model_path = os.path.join(os.getcwd(), '..', 'skintype.keras')
print("-----------------------", model_path)
model = tf.keras.models.load_model(model_path)

@app.route('/')
def skin():
    img = os.path.join(os.getcwd(), '..', '1.jpg')
    test_image = image.load_img(
        img,
        target_size=(64, 64)
    )
    test_image= tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    prediction = ''
    if result[0][0] == 1:
        prediction = '1: AIR'
    elif result[0][1] == 1:
        prediction = '2: EARTH'
    elif result[0][2] == 1:
        prediction = '3: FIRE'
    elif result[0][3] == 1:
        prediction = '4: WATER'
    else: prediction = 'Error'
    print(result)
    print(prediction)
    if prediction!='Error':
        return prediction.split(":")[1]
    else:
        array = np.array(result[0])
        element = max(result[0])
        # Find the index of the element 5 in the array.
        index = np.where(array == element)[0][0]
        print("getting index from maximum value", index)
        if index == 0:
            return 'AIR'
        elif index == 1:
            return 'EARTH'
        elif index == 2:
            return 'FIRE'
        elif index == 3:
            return 'WATER'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))



