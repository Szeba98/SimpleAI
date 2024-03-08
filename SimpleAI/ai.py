import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Betöltjük az előre tanított MobileNet modellt
model = MobileNet(weights='imagenet')
array = ["beagle.jpg","golden.jpg","juhasz.jpg","vizsla.jpg"]

def predict_image(image_path):
    # Betöltjük a képet és preprocesszáljuk
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Előrejelzés generálása a modell segítségével
    predictions = model.predict(img_array)
    predicted_classes = decode_predictions(predictions, top=3)[0]  # Kiválasztjuk a legvalószínűbb 3 osztályt

    # Eredmények kiíratása
    for i, (imagenet_id, label, score) in enumerate(predicted_classes):
        print(f"{i + 1}. {label}: {score:.2f}")

# Teszteljük az előrejelzést egy macska és egy kutya képével

number = 0
for string in array:
    number+=1
    print(f"\n{number}.Teszt Kutya kép előrejelzése:")
    predict_image(string)

