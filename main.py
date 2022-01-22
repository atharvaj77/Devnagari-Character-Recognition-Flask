import base64

import cv2
import numpy as np
from flask import Flask
from flask import render_template, request
from tensorflow import keras

# Creating Flask instance
app = Flask(__name__)

classes = ['character_10_yna',
           'character_11_taamatar',
           'character_12_thaa',
           'character_13_daa',
           'character_14_dhaa',
           'character_15_adna',
           'character_16_tabala',
           'character_17_tha',
           'character_18_da',
           'character_19_dha',
           'character_1_ka',
           'character_20_na',
           'character_21_pa',
           'character_22_pha',
           'character_23_ba',
           'character_24_bha',
           'character_25_ma',
           'character_26_yaw',
           'character_27_ra',
           'character_28_la',
           'character_29_waw',
           'character_2_kha',
           'character_30_motosaw',
           'character_31_petchiryakha',
           'character_32_patalosaw',
           'character_33_ha',
           'character_34_chhya',
           'character_35_tra',
           'character_36_gya',
           'character_3_ga',
           'character_4_gha',
           'character_5_kna',
           'character_6_cha',
           'character_7_chha',
           'character_8_ja',
           'character_9_jha']

img_paths = [
    'static/yna.png',
    'static/taamatar.png',
    'static/thaa.png',
    'static/daa.png',
    'static/dhaa.png',
    'static/adna.png',
    'static/tabala.png',
    'static/tha.png',
    'static/da.png',
    'static/dha.png',
    'static/ka.png',
    'static/na.png',
    'static/pa.png',
    'static/pha.png',
    'static/ba.png',
    'static/bha.png',
    'static/ma.png',
    'static/yaw.png',
    'static/ra.png',
    'static/la.png',
    'static/waw.png',
    'static/kha.png',
    'static/motosaw.png',
    'static/petchiryakha.png',
    'static/patalosaw.png',
    'static/ha.png',
    'static/chhya.png',
    'static/tra.png',
    'static/gya.png',
    'static/ga.png',
    'static/gha.png',
    'static/kna.png',
    'static/cha.png',
    'static/chha.png',
    'static/ja.png',
    'static/jha.png'
]

# Loading the pretrained keras model
model = keras.models.load_model('devnagari_model_final_final.h5')


# Loading the home page
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


# Loading the predict page
@app.route('/', methods=['POST'])
def predict():
    canvasdata = request.form['canvasimg']
    encoded_data = request.form['canvasimg'].split(',')[1]
    # Ref: https://stackoverflow.com/questions/3470546/how-do-you-decode-base64-data-in-python
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (32, 32))
    img = np.expand_dims(gray_image, axis=0)

    try:
        prediction = model.predict(img)
        index = np.argmax(prediction)
        final = classes[index]
        img_path = img_paths[index]

        print(prediction)

        print(f"Prediction Result : {str(final)}")
        return render_template('home.html', response=str(final), canvasdata=canvasdata, img_src=img_path,
                               success=True)
    except Exception as e:
        return render_template('home.html', response=str(e), canvasdata=canvasdata, img_src=img_path)


if __name__ == '__main__':
    app.run(debug=True)
