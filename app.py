import streamlit as st
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from lime import lime_image
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries


all_models = ["NASNetMobile", "ResNet50V2", "ResNet101V2", "Xception"]
target_sizes = dict(zip(all_models, [(x, x) for x in [224, 331, 331, 331]]))
drive_link = "https://drive.google.com/uc?id=1ClDquQj9mTRObaEOwNZCeb47gsFjJK3i"
# drive_link = "https://drive.google.com/uc?id=1hCFAJbY80l27bmA_hlTy6eDvYJ34OWgA"


@st.cache
def download_models():
    gdown.cached_download(drive_link, "model-NASNetMobile.h5", quiet=False)


@st.cache(allow_output_mutation=True)
def get_model(model_name):
    download_models()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return load_model(f"model-{model_name}.h5", custom_objects={'f1': None})


@st.cache
def explain_pred(model, img):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img.astype(np.double), model.predict, hide_color=0, num_samples=1000)
    return explanation


if __name__ == '__main__':
    st.header("Нейросеть для обнаружения заболеваний листьев яблони по фотографии")

    st.sidebar.subheader("Выбор модели")
    model_name = st.sidebar.selectbox("Модель", all_models, index=0)

    if model_name:
        model = get_model(model_name)

    st.sidebar.subheader("Загрузка изображения")
    file = st.sidebar.file_uploader("Загрузить", type="jpg")

    if file:
        st.subheader("Изображение:")
        st.image(file.read(), width=300)
        file.seek(0)
        if model_name:
            model = get_model(model_name)
            img = Image.open(file)
            tgt_size = target_sizes[model_name]
            img = img.resize(tgt_size, Image.ANTIALIAS)
            img_array = image.img_to_array(img)
            prediction = float(model.predict(np.array([img_array]))[0][0])
            print(prediction)
            st.subheader(f"Предсказание: {['не заражен', 'заражен'][int(prediction > 0.2)]}")
            conf = max(prediction, 1-prediction)
            st.write(f"Уверенность: {conf*100:.2f}%")
            explain = st.button("Объяснить предсказание")
            if explain:
                exp = explain_pred(model, img_array)
                dict_heatmap = dict(exp.local_exp[0])
                heatmap = np.vectorize(dict_heatmap.get)(exp.segments)
                fig, ax = plt.subplots()
                ax.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
                ax.axis('off')
                st.pyplot(fig)

    else:
        st.write("Пожалуйста, загрузите изображение.")

