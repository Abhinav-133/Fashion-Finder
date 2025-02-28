import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import time  # For loading animation

# Load trained model
model = tf.keras.models.load_model("../model/outfit_recommendation_model.h5")
feature_extractor_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

# Load dataset features & metadata
dataset_features = np.load("../model/dataset_features.npy")
image_paths = np.load("../model/image_paths.npy")
dataset_categories = np.load("../model/dataset_categories.npy")
dataset_types = np.load("../model/dataset_types.npy")

titles = np.load("../model/titles.npy", allow_pickle=True)
prices = np.load("../model/prices.npy", allow_pickle=True)
links = np.load("../model/links.npy", allow_pickle=True)

# Function to preprocess user-uploaded image
def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0  
    img = np.expand_dims(img, axis=0) 
    return img

# Outfit recommendation function
def recommend_outfit(uploaded_image, top_n=5, sort_by="Similarity"):
    query_img = preprocess_image(uploaded_image)
    
    query_features = feature_extractor_model.predict(query_img)[0]
    query_category, query_type = model.predict(query_img)
    query_category = np.argmax(query_category)
    query_type = np.argmax(query_type)

    similarities = cosine_similarity([query_features], dataset_features)[0]
    valid_indices = [i for i in range(len(dataset_features)) if dataset_categories[i] == query_category and dataset_types[i] == query_type]

    filtered_similarities = [(i, similarities[i]) for i in valid_indices]
    filtered_similarities.sort(key=lambda x: x[1], reverse=True)
    
    recommendations = [{
        "image_path": os.path.abspath(image_paths[i]),
        "title": titles[i],
        "price": prices[i],
        "link": links[i],
        "similarity": score
    } for i, score in filtered_similarities[:top_n]]
    
    if sort_by == "Price: Low to High":
        recommendations.sort(key=lambda x: x["price"])
    elif sort_by == "Price: High to Low":
        recommendations.sort(key=lambda x: x["price"], reverse=True)
    
    return recommendations

# Streamlit UI
st.set_page_config(page_title="Fashion Finder", page_icon="🛍️", layout="wide")

# Custom CSS for better UI
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        /* Set full-page background */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
            color: #fff;
        }

        /* Streamlit main container background */
        [data-testid="stAppViewContainer"] > .main {
            background: transparent !important;
        }

        /* Content box styling */
        .stApp {
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 20px;
            backdrop-filter: blur(12px);
            box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.3);
        }

        /* Title */
        .title {
            font-size: 45px;
            font-weight: bold;
            text-align: center;
            color: white;
            padding: 10px;
            text-shadow: 3px 3px 15px rgba(0, 0, 0, 0.4);
        }

        /* Upload box */
        .upload-box {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        }

        /* Recommendation cards */
        .recommendation-card {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 15px;
            transition: all 0.3s;
        }

        .recommendation-card:hover {
            transform: scale(1.05);
            box-shadow: 0px 8px 25px rgba(255, 255, 255, 0.4);
        }

        /* Buy Now Button */
        .buy-now-link {
            color: white;
            text-decoration: none;
            font-weight: bold;
            padding: 10px 15px;
            border-radius: 10px;
            background: #ff9f1c;
            display: inline-block;
            margin-top: 10px;
            transition: all 0.3s;
        }

        .buy-now-link:hover {
            background: #ff7b00;
            color: white;
            box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.4);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>🛍️ Fashion Finder</div>", unsafe_allow_html=True)
st.write("Upload an outfit image, and we'll recommend **visually similar outfits** with price & buy links!")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_file, caption="📷 Your Uploaded Image", use_column_width=True)

    with col2:
        st.write("🔍 **Finding similar outfits...**")
        with st.spinner("✨ Processing... Please wait"):
            time.sleep(2)  # Simulate loading time
            image = Image.open(uploaded_file)
            sort_option = st.selectbox("Sort By:", ["Price: Low to High", "Price: High to Low"])
            recommendations = recommend_outfit(image, top_n=5, sort_by=sort_option)

        if recommendations:
            st.subheader("🎯 **Recommended Outfits**")
            for rec in recommendations:
                col_img, col_info = st.columns([1, 3])
                
                with col_img:
                    st.image(rec["image_path"], width=200)
                
                with col_info:
                    st.markdown(f"<h4 style='color: #333;'>{rec['title']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #e44d26; font-weight: bold;'>💰 ₹{rec['price']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<a href='{rec['link']}' class='buy-now-link'>🛒 Buy Now</a>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No matching outfits found.")
