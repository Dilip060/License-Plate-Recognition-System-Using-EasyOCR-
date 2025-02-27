# License-Plate-Recognition-System-Using-EasyOCR-
The system consists of two main components: 1. License Plate Detection: Uses EasyOCR to detect license plates and extract bounding box coordinates. 2. Character Recognition: Uses a Convolutional Neural Network (CNN) to recognize the characters on the license plates.
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

# Path to car images
car_images_path = r'C:\Users\user\Downloads\DATA SCIENTIST_ASSIGNMENT-20250226T162947Z-001\DATA SCIENTIST_ASSIGNMENT\test\test\test'

# Load car images
car_images = []
for filename in os.listdir(car_images_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other formats if needed
        image_path = os.path.join(car_images_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            car_images.append(image)

print(f"Loaded {len(car_images)} car images.")
# Path to license plate images
license_plate_images_path = r'C:\Users\user\Downloads\DATA SCIENTIST_ASSIGNMENT-20250226T162947Z-001\DATA SCIENTIST_ASSIGNMENT\Licplatesdetection_train\license_plates_detection_train'

# Load license plate images
license_plate_images = []
for filename in os.listdir(license_plate_images_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other formats if needed
        image_path = os.path.join(license_plate_images_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale for character recognition
        if image is not None:
            license_plate_images.append(image)

print(f"Loaded {len(license_plate_images)} license plate images.")
def preprocess_car_image(image, target_size=(128, 128)):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

preprocessed_car_images = [preprocess_car_image(img) for img in car_images]
def preprocess_license_plate(image, target_size=(64, 64)):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

preprocessed_license_plates = [preprocess_license_plate(img) for img in license_plate_images]
import easyocr
import cv2
import os
import pandas as pd

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Path to the folder containing car images
car_images_path = r'C:\Users\user\Downloads\DATA SCIENTIST_ASSIGNMENT-20250226T162947Z-001\DATA SCIENTIST_ASSIGNMENT\test\test\test'

# Output CSV file to save annotations
output_csv = 'automatic_annotations.csv'

# Write headers to the CSV file
with open(output_csv, 'w') as file:
    file.write("filename,xmin,ymin,xmax,ymax,text,confidence\n")

# Loop through each image in the folder
for filename in os.listdir(car_images_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(car_images_path, filename)
        print(f"Processing {image_path}...")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            continue

        # Perform text detection
        results = reader.readtext(image)

        # Save detected text and bounding boxes to CSV
        with open(output_csv, 'a') as file:
            for result in results:
                bbox, text, confidence = result
                xmin, ymin = map(int, bbox[0])
                xmax, ymax = map(int, bbox[2])
                file.write(f"{filename},{xmin},{ymin},{xmax},{ymax},{text},{confidence}\n")

print("Automatic annotation process completed.")
import os
print(os.getcwd())
import os
import cv2
import numpy as np
import pandas as pd

# Load the annotations
annotations = pd.read_csv('automatic_annotations.csv')

# Path to the folder containing car images
car_images_path = r'C:\Users\user\Downloads\DATA SCIENTIST_ASSIGNMENT-20250226T162947Z-001\DATA SCIENTIST_ASSIGNMENT\test\test\test'

# Path to save cropped license plate images
output_folder = 'cropped_license_plates'
os.makedirs(output_folder, exist_ok=True)

# Preprocess and save cropped license plates
cropped_images = []
text_labels = []

for index, row in annotations.iterrows():
    filename = row['filename']
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    text = row['text']

    # Load the image
    image_path = os.path.join(car_images_path, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        continue

    # Get image dimensions
    height, width, _ = image.shape

    # Validate bounding box coordinates
    if xmin >= xmax or ymin >= ymax:
        print(f"Invalid bounding box for {filename}: ({xmin}, {ymin}, {xmax}, {ymax})")
        continue
    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        print(f"Bounding box out of bounds for {filename}: ({xmin}, {ymin}, {xmax}, {ymax})")
        continue

    # Crop the license plate
    license_plate = image[ymin:ymax, xmin:xmax]

    # Check if the cropped image is valid
    if license_plate.size == 0:
        print(f"Empty license plate for {filename}: ({xmin}, {ymin}, {xmax}, {ymax})")
        continue

    # Resize and normalize the license plate
    license_plate = cv2.resize(license_plate, (128, 64))  # Adjust size as needed
    license_plate = license_plate / 255.0  # Normalize to [0, 1]

    # Save the cropped license plate
    output_path = os.path.join(output_folder, f"cropped_{filename}")
    cv2.imwrite(output_path, license_plate * 255)  # Save as 8-bit image

    # Append to lists
    cropped_images.append(license_plate)
    text_labels.append(text)

# Convert to numpy arrays
cropped_images = np.array(cropped_images)
text_labels = np.array(text_labels)

print(f"Preprocessed {len(cropped_images)} license plates.")
import os
import cv2
import numpy as np
import pandas as pd

# Load the annotations
annotations = pd.read_csv('automatic_annotations.csv')

# Path to the folder containing car images
car_images_path = r'C:\Users\user\Downloads\DATA SCIENTIST_ASSIGNMENT-20250226T162947Z-001\DATA SCIENTIST_ASSIGNMENT\test\test\test'

# Path to save cropped license plate images
output_folder = 'cropped_license_plates'
os.makedirs(output_folder, exist_ok=True)

# Preprocess and save cropped license plates
cropped_images = []
text_labels = []

for index, row in annotations.iterrows():
    filename = row['filename']
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    text = row['text']

    # Load the image
    image_path = os.path.join(car_images_path, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        continue

    # Get image dimensions
    height, width, _ = image.shape

    # Validate bounding box coordinates
    if xmin >= xmax or ymin >= ymax:
        print(f"Invalid bounding box for {filename}: ({xmin}, {ymin}, {xmax}, {ymax})")
        continue
    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        print(f"Bounding box out of bounds for {filename}: ({xmin}, {ymin}, {xmax}, {ymax})")
        continue

    # Crop the license plate
    license_plate = image[ymin:ymax, xmin:xmax]

    # Check if the cropped image is valid
    if license_plate.size == 0:
        print(f"Empty license plate for {filename}: ({xmin}, {ymin}, {xmax}, {ymax})")
        continue

    # Resize and normalize the license plate
    license_plate = cv2.resize(license_plate, (128, 64))  # Adjust size as needed
    license_plate = license_plate / 255.0  # Normalize to [0, 1]

    # Save the cropped license plate
    output_path = os.path.join(output_folder, f"cropped_{filename}")
    cv2.imwrite(output_path, license_plate * 255)  # Save as 8-bit image

    # Append to lists
    cropped_images.append(license_plate)
    text_labels.append(text)

# Convert to numpy arrays
cropped_images = np.array(cropped_images)
text_labels = np.array(text_labels)

print(f"Preprocessed {len(cropped_images)} license plates.")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode text labels into numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(text_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cropped_images, encoded_labels, test_size=0.2, random_state=42)

# Define the CNN model
def create_recognition_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
input_shape = (64, 128, 3)  # Adjust based on your image size
num_classes = len(label_encoder.classes_)
recognition_model = create_recognition_model(input_shape, num_classes)
recognition_model.summary()

# Train the model
recognition_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# Evaluate the model
test_loss, test_accuracy = recognition_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on a sample image
sample_image = X_test[0]
predicted_label = recognition_model.predict(np.expand_dims(sample_image, axis=0))
predicted_text = label_encoder.inverse_transform([np.argmax(predicted_label)])[0]
print(f"Predicted Text: {predicted_text}")
# Save the model
recognition_model.save('license_plate_recognition_model.h5')
print("Model saved as 'license_plate_recognition_model.h5'.")
# Load the saved model
model = tf.keras.models.load_model('license_plate_recognition_model.h5')

# Function to recognize license plate text
def recognize_license_plate(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 64))  # Resize to match model input
    image = image / 255.0  # Normalize

    # Predict the text
    predicted_label = model.predict(np.expand_dims(image, axis=0))
    predicted_text = label_encoder.inverse_transform([np.argmax(predicted_label)])[0]
    return predicted_text

# Test on a new image
new_image_path = r'C:\Users\user\Desktop\licplt new.jpeg'
predicted_text = recognize_license_plate(new_image_path)
print(f"Predicted License Plate Text: {predicted_text}")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode text labels into numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(text_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cropped_images, encoded_labels, test_size=0.2, random_state=42)

# Define the CNN model
def create_recognition_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
input_shape = (64, 128, 3)  # Adjust based on your image size
num_classes = len(label_encoder.classes_)
recognition_model = create_recognition_model(input_shape, num_classes)
recognition_model.summary()

# Train the model
recognition_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Step 1: Load the annotations
annotations = pd.read_csv('automatic_annotations.csv')

# Step 2: Preprocess the data
car_images_path = r'C:\Users\user\Downloads\DATA SCIENTIST_ASSIGNMENT-20250226T162947Z-001\DATA SCIENTIST_ASSIGNMENT\test\test\test'
output_folder = 'cropped_license_plates'
os.makedirs(output_folder, exist_ok=True)

cropped_images = []
text_labels = []

for index, row in annotations.iterrows():
    filename = row['filename']
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    text = row['text']

    # Load the image
    image_path = os.path.join(car_images_path, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        continue

    # Validate bounding box coordinates
    height, width, _ = image.shape
    if xmin >= xmax or ymin >= ymax:
        print(f"Invalid bounding box for {filename}: ({xmin}, {ymin}, {xmax}, {ymax})")
        continue
    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        print(f"Bounding box out of bounds for {filename}: ({xmin}, {ymin}, {xmax}, {ymax})")
        continue

    # Crop the license plate
    license_plate = image[ymin:ymax, xmin:xmax]
    if license_plate.size == 0:
        print(f"Empty license plate for {filename}: ({xmin}, {ymin}, {xmax}, {ymax})")
        continue

    # Resize and normalize the license plate
    license_plate = cv2.resize(license_plate, (128, 64))  # Resize to match model input
    license_plate = license_plate / 255.0  # Normalize to [0, 1]

    # Save the cropped license plate
    output_path = os.path.join(output_folder, f"cropped_{filename}")
    cv2.imwrite(output_path, license_plate * 255)  # Save as 8-bit image

    # Append to lists
    cropped_images.append(license_plate)
    text_labels.append(text)

# Convert to numpy arrays
cropped_images = np.array(cropped_images)
text_labels = np.array(text_labels)

print(f"Preprocessed {len(cropped_images)} license plates.")

# Step 3: Encode text labels into numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(text_labels)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cropped_images, encoded_labels, test_size=0.2, random_state=42)

# Step 5: Define the CNN model
def create_recognition_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 6: Create and train the model
input_shape = (64, 128, 3)  # Adjust based on your image size
num_classes = len(label_encoder.classes_)
recognition_model = create_recognition_model(input_shape, num_classes)
recognition_model.summary()

# Train the model
recognition_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 7: Save the model
recognition_model.save('license_plate_recognition_model.h5')
print("Model saved as 'license_plate_recognition_model.h5'.")
# Load the saved model
model = tf.keras.models.load_model('license_plate_recognition_model.h5')

# Function to recognize license plate text
def recognize_license_plate(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # Resize and normalize the image
    image = cv2.resize(image, (128, 64))  # Resize to match model input
    image = image / 255.0  # Normalize

    # Predict the text
    predicted_label = model.predict(np.expand_dims(image, axis=0))
    predicted_text = label_encoder.inverse_transform([np.argmax(predicted_label)])[0]
    return predicted_text

# Test on a new image
new_image_path = r'C:\Users\user\Desktop\licplt new.jpeg'
predicted_text = recognize_license_plate(new_image_path)
if predicted_text is not None:
    print(f"Predicted License Plate Text: {predicted_text}")
# Load the saved model
model = tf.keras.models.load_model('license_plate_recognition_model.h5')

# Function to recognize license plate text
def recognize_license_plate(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # Resize and normalize the image
    image = cv2.resize(image, (128, 64))  # Resize to match model input
    image = image / 255.0  # Normalize

    # Predict the text
    predicted_label = model.predict(np.expand_dims(image, axis=0))
    predicted_text = label_encoder.inverse_transform([np.argmax(predicted_label)])[0]

    # Post-process the predicted text
    predicted_text = predicted_text.replace(" ", "")  # Remove spaces
    return predicted_text

# Test on a new image
new_image_path = r'C:\Users\user\Desktop\licplt2.jpg'
predicted_text = recognize_license_plate(new_image_path)
if predicted_text is not None:
    print(f"Predicted License Plate Text: {predicted_text}")
# Save the cropped license plate for inspection
cv2.imwrite('cropped_license_plate.jpg', image * 255)
print("Label Encoder Classes:", label_encoder.classes_)
