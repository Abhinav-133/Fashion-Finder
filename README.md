# Outfit Recommendation System

The Outfit Recommendation System is an end-to-end application designed to assist users in discovering and refining their style. It uses web scraping to gather outfit data, extensive preprocessing to prepare images, a custom CNN model for outfit classification, OpenCV for image processing, and a recommendation engine to suggest complementary pieces.

---

## Overview

This project integrates multiple components to deliver a robust outfit recommendation solution:

- **Web Scraping:** Extracts outfit data and related fashion content from various online sources.
- **Preprocessing:** Processes images using OpenCV and Python libraries for accurate classification.
- **Deep Learning:** Employs a custom Convolutional Neural Network (CNN) to classify outfits into specific categories.
- **Recommendation Engine:** Uses similarity metrics and classification results to generate tailored outfit recommendations.
- **User Interface:** Interactive front-end built with Streamlit to provide an intuitive user experience.

---

## Features

- **Web Scraping:** Automatically collects data and images from fashion websites to maintain updated fashion trends.
- **Image Upload:** Users can upload outfit images to receive recommendations.
- **Preprocessing Pipeline:** Cleans and prepares images using OpenCV, ensuring they are optimized for the CNN model.
- **CNN-Based Classification:** Utilizes a custom-trained CNN to categorize outfits into predefined styles.
- **Real-Time Recommendations:** Computes recommendations on-the-fly with an efficient backend.
- **Interactive UI:** Streamlit-based interface for simple navigation and interaction.
- **Detailed Analytics:** Provides insights into classification performance and recommendation accuracy.

---

## Technologies

- **Python:** Core language used for development.
- **TensorFlow/Keras or PyTorch:** For building and training the CNN model.
- **OpenCV:** For image processing and preprocessing tasks.
- **BeautifulSoup / Scrapy:** For web scraping to collect relevant outfit data.
- **Pandas & Numpy:** For data manipulation and analysis.
- **Streamlit:** For building an interactive and responsive user interface.

---

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/outfit-recommendation-system.git
    cd outfit-recommendation-system
    ```

2. **Create and Activate a Virtual Environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    ```

3. **Install the Required Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Data and Model Files:**

    - **Dataset:** Place your collected and/or scraped dataset in the `data/` directory.
    - **Models:** Place any pre-trained model weights in the `models/` directory.

---

## Usage

1. **Run the Application:**

    ```bash
    streamlit run app.py
    ```

2. **Using the Interface:**

    - **Upload an Image:** Use the provided interface to upload an outfit photo.
    - **View Recommendations:** Check the recommended outfit pieces with classification details.
    - **Explore Data:** Analyze model performance and processed data through the dashboard (if implemented).

---
## Dataset, Training, and Preprocessing

- **Data Collection:**
  - Data is gathered using web scraping techniques from various fashion websites using tools such as BeautifulSoup or Scrapy. This allows the system to continuously update with the latest fashion trends.
  - The scraped images and metadata are stored in the `data/` directory.

- **Preprocessing Pipeline:**
  - **Image Cleaning & Resizing:**  
    Images are cleaned to remove noise and resized to a standard shape to ensure consistency during model training.  
    *Example:*  
    ```python
    import cv2

    def preprocess_image(image_path, target_size=(224, 224)):
        image = cv2.imread(image_path)
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    ```
  - **Normalization & Augmentation:**  
    Before training, images undergo normalization (to scale pixel values) and augmentation (e.g., rotations, flips) to improve the model’s robustness.
  - **Labeling:**  
    Each image is associated with a label representing its outfit category, which is essential for supervised training.

- **Training the CNN Model:**
  - The CNN is built using frameworks like TensorFlow/Keras or PyTorch.  
  - Training involves splitting the dataset into training, validation, and testing sets.
  - *Example Training Command:*
    ```bash
    python src/train_model.py --epochs 25 --batch-size 32
    ```
  - The training script handles data loading, preprocessing, model training, and saving the trained model in the `models/` directory.

---

## Architecture

- **Frontend (User Interface):**  
  - Developed using Streamlit to create an interactive experience.
  - Users can upload outfit images and view the recommendations in real time.

- **Backend Modules:**
  - **Web Scraping Module:**  
    - Continuously gathers data from fashion websites.
    - Processes and stores images along with relevant metadata.
  - **Preprocessing Module:**  
    - Uses OpenCV and other Python libraries to clean, resize, and augment images for training.
  - **CNN Model Module:**  
    - A custom-built Convolutional Neural Network (CNN) classifies outfits into pre-defined categories.
    - The model architecture is defined in `src/model.py`, and training is handled by `src/train_model.py`.
  - **Recommendation Engine:**  
    - Combines output from the CNN with similarity metrics to generate personalized outfit suggestions.
    - The recommendation logic is implemented in `src/recommend.py`.

- **Data Flow Overview:**
  1. **Data Ingestion:**  
     Images and metadata are scraped and stored.
  2. **Preprocessing:**  
     Images are cleaned, normalized, and augmented.
  3. **Model Training & Inference:**  
     The CNN is trained on the processed dataset and used to classify new images.
  4. **Recommendation:**  
     Based on classification outputs and similarity measures, the engine provides complementary outfit suggestions.
