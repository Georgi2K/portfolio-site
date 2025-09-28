# 🐾 Dog vs Cat Classifier

📌 **Description**  
This project is a simple computer vision pipeline that uses deep learning to classify images as **Dog** or **Cat**.  
It consists of two main parts:  
1. **Model Training** – builds and trains a neural network (`image_classifier.h5`) on a dataset of dog and cat images.  
2. **Model Evaluation & Prediction** – loads the trained model to test accuracy and classify new images.  

The dataset already includes a collection of **dog** and **cat** images for training and testing.  
👉 You’re always free to replace the dataset and change the classification topic (e.g., **Cars vs Planes**, **Apples vs Oranges**, etc.).  

**Technologies**  
- Python 
- TensorFlow / Keras  
- Matplotlib (for preprocessing & visualization)  

**Functionality**  
- **Dataset Handling**: Load and preprocess dog/cat images.  
- **Model Training**: Train a CNN and save it as `image_classifier.h5`.  
- **Model Evaluation**: Calculate accuracy on validation/test sets.  
- **Prediction**: Given a new image, determine if it’s a **Dog** 🐶 or a **Cat** 🐱.  

---

🚀 **Usage**  
1. Train the model:  
2. Put in your own downloaded image: (switch it up with the image i've put in on: Model_Training2FIX Line 52, image_path)
3. Start the 2nd model
4. Check if the computer was right!
