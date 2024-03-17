import numpy as np
import tensorflow as tf
import os
from keras.preprocessing import image

normal_model = tf.keras.models.load_model('./Model/model_without_image_gen.keras')
gen_model = tf.keras.models.load_model('./Model/model_with_image_gen.keras')

predict_files = os.listdir('./Predict')
for file in predict_files:
    img = image.load_img('./Predict/' + file, target_size=(150, 150))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    pred_normal = normal_model.predict(images, batch_size=10)
    pred_better = gen_model.predict(images, batch_size=10)
    print(pred_normal[0][0])
    print(pred_better[0][0])
    print("< 일반 모델 판단 결과 >")
    if pred_normal[0][0] > 0:
        print(file + "은 돌멩이")
    else:
        print(file + "은 뗀석기")
    print()
    print("< 개선 모델 판단 결과 >")
    if pred_better[0][0] > 0:
        print(file + "은 돌멩이")
    else:
        print(file + "은 뗀석기")
    print()
    print('---------------------------------')
