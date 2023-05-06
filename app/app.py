import base64

import streamlit as st
import numpy as np

from PIL import Image
from skimage import color

from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

MODEL = load_model("../model/model.h5")

RESIZE_IMG = lambda x: x.resize((300, 300))

def predict(pil_img):
    image_size = (512,512)
    pil_img = pil_img.resize(image_size)
    
    np_img = np.array(pil_img).reshape(1,512, 512, 1)
    
    probs = MODEL.predict(np_img)
    pred = (probs >= 0.7).astype(np.uint8)
    mask_sq = np.squeeze(pred)
    ones = np.ones((512, 512))
    conf = cosine_similarity(mask_sq, ones).mean()
    
    result_image = color.label2rgb(pred[0,...,0], np_img[0,...,0], colors=[[0.9,0.2,0.2]], alpha=0.4)
    
    result_image = np.rot90((result_image*255).astype(np.uint8))
    
    return result_image, 1-conf

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

top_ways = """<p style="font-family:sans-serif; color:White; font-size: 21px;">Top 5 ways to keep your liver healthy:</p>
            <ol style="font-family:sans-serif; color:White; font-size: 16px;">
            <li><span style="background-color:rgba(0,0,0,.5);">Be careful about alcohol consumption</span></li>
            <li><span style="background-color:rgba(0,0,0,.5);">Wash produce and steer clear of toxins</span></li>
            <li><span style="background-color:rgba(0,0,0,.5);">Prevent hepatitis A, B and C</span></li>
            <li><span style="background-color:rgba(0,0,0,.5);">Watch out for medications and herbs</span></li>
            <li><span style="background-color:rgba(0,0,0,.5);">Exercise and eat right</span></li>
            </ol>
                <p style="font-family:sans-serif; color:White; font-size: 11px;">
                <a href="https://www.hopkinsmedicine.org/health/wellness-and-prevention/5-ways-to-be-kind-to-your-liver">Source: Hopkins Medicine</a></p>
                """

about = """<h2>About</h2>
            <p>The objective of this project is to automatically delineate liver on patient scans
            by computer vision. The data used consists of 20 medical examinations in 3D MRIs. Model used
            for training: U-Net with Dice Score = 94.6%</p>"""

if __name__ == "__main__":
    add_bg_from_local("assets/liver.png")
    new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Liver Segmentation</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown(top_ways, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown(about, unsafe_allow_html=True)
        
    uploaded_file = st.file_uploader("Choose a file")
    col1, col2 = st.columns(2)
    with st.container():
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            with col1:
                pil_img = img.rotate(90, expand=True)  
                st_img = RESIZE_IMG(pil_img)
                new_title = '<p style="font-family:sans-serif; color:White; font-size: 21px;">Input</p>'
                st.markdown(new_title, unsafe_allow_html=True)
                st.image(st_img)

                if st.button("Predict"):
                    result, conf = predict(img)
                    with col2:
                        new_title = '<p style="font-family:sans-serif; color:White; font-size: 21px;">Segmentation</p>'
                        st.markdown(new_title, unsafe_allow_html=True)
                        pil_res = Image.fromarray(result, "RGB")
                        st_res = RESIZE_IMG(pil_res)
                        st.image(st_res)
                        confidence = f'<span style="background-color:rgba(0,0,0,.5);">Confidence = {round(conf*100,2)}%</span>'
                        st.markdown(confidence, unsafe_allow_html=True)