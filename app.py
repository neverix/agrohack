import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os


all_models = ["NASNetMobile", "ResNet50V2", "ResNet101V2", "Xception"]
target_sizes = dict(zip(all_models, [(x, x) for x in [224, 331, 331, 331]]))


@st.cache(allow_output_mutation=True)
def get_model(model_name):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return load_model(f"model-{model_name}.h5", custom_objects={'f1': None})


if __name__ == '__main__':
    st.header("Нейросеть для обнаружения заболеваний листьев яблони по фотографии")

    st.sidebar.subheader("Выбор модели")
    model_name = st.sidebar.selectbox("Модель", all_models)

    if model_name:
        model = get_model(model_name)

    st.sidebar.subheader("Загрузка изображения")
    file = st.sidebar.file_uploader("Загрузить", type="jpg")

    if file:
        st.subheader("Изображение:")
        st.image(file.read(), use_column_width=True)
        file.seek(0)
        if model_name:
            model = get_model(model_name)
            img = Image.open(file)
            tgt_size = target_sizes[model_name]
            img = img.resize(tgt_size, Image.ANTIALIAS)
            img_array = image.img_to_array(img)
            prediction = float(model.predict(np.array([img_array]))[0][0])
            st.subheader(f"Предсказание: {['не заражен', 'заражен'][int(round(prediction))]}")
            conf = max(prediction, 1-prediction)
            st.write(f"Уверенность: {conf*100:.2f}%")
    else:
        st.write("Пожалуйста, загрузите изображение.")

