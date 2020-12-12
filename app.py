import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model


model_name = "model.h5"


@st.cache
def get_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return load_model(model_name)


if __name__ == '__main__':
    st.header("Нейросеть для обнаружения заболеваний листьев яблони по фотографии")

    st.subheader("Загрузить изображение")
    file = st.file_uploader("Загрузить", type="jpg")
    st.image(file)

    st.subheader(f"Предсказание: ")

