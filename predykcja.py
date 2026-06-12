import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


MODEL_PATH = 'cnn_cgr_best_kfold_model.keras'

CLASS_NAMES = ['Canidae', 'Cetacea', 'Elephantidae', 'Felidae', 
               'Marsupialia', 'Primates', 'Rodentia', 'Vespertilionidae']

def predict_animal_group(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Błąd: Nie znaleziono pliku modelu '{MODEL_PATH}'!")
        return

    model = load_model(MODEL_PATH)

    img = load_img(image_path, color_mode='grayscale', target_size=(64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) 

    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]
    best_class_idx = np.argmax(score)
    confidence = score[best_class_idx] * 100

    print(f"\n" + "="*30)
    
    print(f"WYNIK ANALIZY DNA")
    print(f"="*30)
    print(f"Plik: {os.path.basename(image_path)}")
    print(f"Przewidziana grupa: {CLASS_NAMES[best_class_idx]}")
    print(f"Pewność modelu: {confidence:.2f}%")
    print("-"*30)
    
    print("Prawdopodobieństwa dla wszystkich klas:")
    for name, prob in zip(CLASS_NAMES, score):
        print(f" • {name:15}: {prob*100:6.2f}%")



obrazek_do_testu = "CGRpng/NC_005089.1.png" 

if __name__ == "__main__":
    predict_animal_group(obrazek_do_testu)