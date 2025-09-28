from PIL import Image
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Adjust these paths
DATASET_PATH = r"C:\Users\Anwender\PycharmProjects\Module7\dataset"
MODEL_PATH = r"C:\Users\Anwender\PycharmProjects\Module7\image_classifier.h5"

num_classes = len([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
class_mode = "binary" if num_classes == 2 else "categorical"


def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at path: {image_path}")
        return None

    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f"Error: Corrupted image - {image_path}")
        return None

    model = tf.keras.models.load_model(MODEL_PATH)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to read image - {image_path}")
        return None

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_names = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

    if class_mode == "binary":
        predicted_class = class_names[int(prediction[0] > 0.5)]
    else:
        predicted_class = class_names[tf.argmax(prediction, axis=-1).numpy()[0]]

    return predicted_class


# Show result
image_path = r"C:\Users\Anwender\PycharmProjects\Module7\dataset\pexels-chevanon-1108099.jpg"
predicted = predict_image(image_path)

if predicted:
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"The Model has determined: {predicted}")
    plt.axis('off')
    plt.show()
    print(f"The Model has determined: {predicted}")