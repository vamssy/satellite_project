import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model('satellite_classifier.h5')
img_path = 'path/to/test_image.jpg'  # Update with your image path
img = load_img(img_path, target_size=(150, 150))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
classes = ['cloudy', 'desert', 'green_area', 'water']
predicted_class = classes[np.argmax(prediction)]
print(f"Predicted class: {predicted_class}")