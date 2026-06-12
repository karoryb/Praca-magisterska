import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


tf.get_logger().setLevel('ERROR') 


CGR_PNG_DIR = "CGRpng" 
IMG_SIZE = 64 
CHANNELS = 1 
EPOCHS = 100
N_SPLITS = 5 
BATCH_SIZE = 32
MIN_SAMPLES_PER_CLASS = N_SPLITS 



def get_accession_to_class_map():
    """
    Zwraca słownik mapujący Accession ID na klasy taksonomiczne.
    """
    return {
        'NC_005089.1': "Rodentia",
        'NC_012374.1': "Rodentia",
        'NC_000884.1': "Rodentia",
        'NC_037509.1': "Rodentia",
        'NC_034314.1': "Rodentia",
        'NC_027684.1': "Rodentia",
        'NC_027683.1': "Rodentia",
        'NC_048490.1': "Rodentia",
        'NC_018367.1': "Rodentia",
        'NC_025902.1': "Rodentia",
        'NC_025316.1': "Rodentia",
        'NC_023780.1': "Rodentia",
        'NC_053822.1': "Rodentia",
        'NC_031802.1': "Rodentia",
        'NC_001913.1': "Rodentia",
        'NC_011120.1': "Primates",
        'NC_011137.1': "Primates",
        'NC_012920.1': "Primates",
        'NC_002082.1': "Primates",
        'NC_014042.1': "Primates",
        'NC_001643.1': "Primates",
        'Y18001.1': "Primates",
        'NC_020009.2': "Primates",
        'NC_002083.1': "Primates",
        'NC_014047.1': "Primates",
        'MT711860.1': "Primates",
        'NC_028306.1': "Felidae",
        'NC_001700.1': "Felidae",
        'OR095102.1': "Felidae",
        'OR095103.1': "Felidae",
        'NC_027083.1': "Felidae",
        'NC_022842.1': "Felidae",
        'NC_010642.1': "Felidae",
        'OR777682.1': "Felidae",
        'MW257216.1': "Felidae",
        'NC_016470.1': "Felidae",
        'NC_026529.1': "Canidae",
        'NC_008434.1': "Canidae",
        'AY729880.1': "Canidae",
        'NC_067757.1': "Canidae",
        'NC_009686.1': "Canidae",
        'NC_013700.1': "Canidae",
        'NC_028427.1': "Canidae",
        'NC_036369.1': "Canidae",
        'NC_008093.1': "Canidae",
        'NC_013445.1': "Canidae",
        'NC_019591.1': "Cetacea",
        'NC_019590.1': "Cetacea", 
        'NC_019589.1': "Cetacea",
        'NC_019588.1': "Cetacea",
        'NC_064558.1': "Cetacea",
        'NC_012059.1': "Cetacea",
        'NC_012058.1': "Cetacea",
        'NC_019577.1': "Cetacea",
        'NC_019578.2': "Cetacea",
        'NC_083222.1': "Cetacea",
        'NC_083094.1': "Cetacea",
        'NC_012062.1': "Cetacea",
        'NC_012051.1': "Cetacea",
        'NC_012053.1': "Cetacea",
        'NC_012061.1': "Cetacea",
        'NC_012057.1': "Cetacea",
        'NC_083148.1': "Vespertilionidae",
        'OR490498.1': "Vespertilionidae",
        'NC_082320.1': "Vespertilionidae",
        'PQ064114.1': "Vespertilionidae",
        'NC_057092.1': "Vespertilionidae",
        'NC_056111.1': "Vespertilionidae",
        'NC_041638.1': "Vespertilionidae",
        'NC_041160.1': "Vespertilionidae",
        'NC_027237.1': "Vespertilionidae",
        'NC_034227.1': "Vespertilionidae",
        'NC_029849.1': "Vespertilionidae",
        'NC_029422.1': "Vespertilionidae",
        'NC_029346.1': "Vespertilionidae",
        'NC_029191.1': "Vespertilionidae",
        'MZ457524.1': "Vespertilionidae",
        'NC_007596.2': "Elephantidae",
        'NC_015529.1': "Elephantidae",
        'NC_005129.2': "Elephantidae",
        'NC_000934.1': "Elephantidae",
        'OL628830.1': "Elephantidae",
        'JN673263.1': "Elephantidae",
        'MK360903.1': "Marsupialia",
        'NC_001794.1': "Marsupialia", 
        'NC_061372.1': "Marsupialia",
        'NC_069653.1': "Marsupialia",
        'NC_057520.1': "Marsupialia",
        'NC_057519.1': "Marsupialia",
        'NC_008133.1': "Marsupialia",
        'NC_018788.1': "Marsupialia",
        'NC_039717.1': "Marsupialia",
        'NC_011944.1': "Marsupialia",
        'NC_008447.1': "Marsupialia",
        'NC_008145.1': "Marsupialia",
        'NC_007631.1': "Marsupialia",
        'FJ515782.1': "Marsupialia",
        'KY996500.1':"Marsupialia"
    }



def get_class_label(filename):
    """
    Ekstrahuje numer dostępowy z nazwy pliku i przypisuje klasę.
    """
    ACCESSION_TO_CLASS = get_accession_to_class_map()
    base_name = os.path.splitext(filename)[0]

    main_id_with_version = re.sub(r'_VAR\d+$', '', base_name)
    

    label = ACCESSION_TO_CLASS.get(main_id_with_version)
    if label:
        return label
    

    main_id_no_version = main_id_with_version.split('.')[0]

    for key, value in ACCESSION_TO_CLASS.items():
        if key.startswith(main_id_no_version + '.'):
             return value

    return 'Nieznana'


def show_learning(history, figsize=(12, 4), title_suffix=""):
    """
    Rysuje wykresy krzywych straty i dokładności dla treningu i walidacji.
    """
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='Strata Treningowa (Loss)')
    plt.plot(val_loss, linestyle='dashed', label='Strata Walidacyjna (Val Loss)')
    plt.title(f'Krzywe Straty (Loss) {title_suffix}')
    plt.ylabel('Wartość Straty')
    plt.xlabel('Epoka')
    plt.legend()
    plt.grid(linestyle='dotted')

    plt.subplot(1, 2, 2)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(acc, label='Dokładność Treningowa (Accuracy)')
    plt.plot(val_acc, linestyle='dashed', label='Dokładność Walidacyjna (Val Accuracy)')
    plt.title(f'Krzywe Dokładności (Accuracy) {title_suffix}')
    plt.ylabel('Dokładność')
    plt.xlabel('Epoka')
    plt.legend()
    plt.grid(linestyle='dotted')

    plt.tight_layout()
    plt.show()


def load_data_from_images():
    """
    Wczytuje obrazy PNG, normalizuje je i konwertuje etykiety na format one-hot.
    """
    print("--- 1. Wczytywanie i etykietowanie obrazów CGR ---")
    
    image_list = []
    label_names = []
    
    if not os.path.exists(CGR_PNG_DIR):
        print(f"BŁĄD: Folder '{CGR_PNG_DIR}' nie istnieje. Utwórz folder i umieść w nim pliki PNG.")
        return None, None, None

    for filename in os.listdir(CGR_PNG_DIR):
        if filename.endswith(".png"):
            label = get_class_label(filename)
            
            if label != 'Nieznana':
                
                filepath = os.path.join(CGR_PNG_DIR, filename)
                
                try:
                    color_mode = 'grayscale' if CHANNELS == 1 else 'rgb'
                    img = load_img(filepath, color_mode=color_mode, target_size=(IMG_SIZE, IMG_SIZE))
                    img_array = img_to_array(img) / 255.0
                    
                    if CHANNELS == 1 and img_array.ndim == 2:
                        img_array = np.expand_dims(img_array, axis=-1)
                    elif CHANNELS == 1 and img_array.shape[-1] == 3:

                        img = load_img(filepath, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
                        img_array = img_to_array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=-1)

                    image_list.append(img_array)
                    label_names.append(label)
                    
                except Exception as e:
                    print(f"Błąd wczytywania/przetwarzania pliku {filename}: {e}")
            else:
                 print(f"Plik '{filename}' pominięty - nie znaleziono mapowania klasy.")
    
    print(f"\nWczytano {len(image_list)} obrazów.")
    
    if not image_list:
        print("Nie wczytano żadnych obrazów. Sprawdź nazwy plików i mapowanie Accession ID.")
        return None, None, None

    X = np.array(image_list)
    y_raw = np.array(label_names)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    class_names = le.classes_
    
    print(f"Wykryte klasy ({len(class_names)}): {class_names}")


    class_counts = pd.Series(y_raw).value_counts()
    for class_name, count in class_counts.items():
        if count < MIN_SAMPLES_PER_CLASS:
            print(f" BŁĄD: Klasa '{class_name}' ma tylko {count} próbek. Walidacja K-Fold ({N_SPLITS} splitów) może wymagać co najmniej {MIN_SAMPLES_PER_CLASS} próbek na klasę.")
            return None, None, None

    
    return X, y_encoded, class_names


# BUDOWA MODELU CNN 

def build_cnn_model(input_shape, num_classes):
    """
    Buduje Konwolucyjną Sieć Neuronową (CNN).
    """
    model = Sequential([
        # Warstwa Konwolucyjna 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)), 

        # Warstwa Konwolucyjna 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # Warstwa Konwolucyjna 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # spłaszczanie danych
        Flatten(),

        # Warstwa Gęsta (Lekko zmniejszona)
        Dense(64, activation='relu'), 
        Dropout(0.4), 

        # Warstwa Wyjściowa
        Dense(num_classes, activation='softmax') 
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def train_and_evaluate_cnn_kfold(X, y_encoded, class_names):
    """
    Przeprowadza walidację krzyżową (K-Fold) dla trenowania i oceny modelu CNN.
    """
    print("\n--- 2. Trening i Ocena Modelu CNN za pomocą K-Fold ---")
    print(f"Liczba foldów: {N_SPLITS}")
    

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    fold_results = []
    all_y_true = np.array([])
    all_y_pred = np.array([])
    best_accuracy = 0
    best_model = None
    
    input_shape = X.shape[1:]
    num_classes = len(class_names)
    
    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, 
        zoom_range=0.1, horizontal_flip=True
    )

    for fold, (train_index, test_index) in enumerate(kf.split(X, y_encoded)):
        print(f"\n--- FOLD {fold+1}/{N_SPLITS} ---")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train_enc, y_test_enc = y_encoded[train_index], y_encoded[test_index]
        
        y_train = to_categorical(y_train_enc, num_classes=num_classes)
        y_test = to_categorical(y_test_enc, num_classes=num_classes)
        
        model = build_cnn_model(input_shape, num_classes)
        
        train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
        
        try:
            history = model.fit(
                train_generator, 
                epochs=EPOCHS,
                steps_per_epoch=len(train_generator),
                validation_data=(X_test, y_test), 
                verbose=2 
            )
        except Exception as e:
            print(f"BŁĄD podczas trenowania w Fold {fold+1}: {e}")
            continue

        
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f" Dokładność na zbiorze testowym foldu {fold+1}: {accuracy:.4f} (Loss: {loss:.4f})")
        
        fold_results.append(accuracy)
        
        y_pred_probs = model.predict(X_test)
        y_pred_fold = np.argmax(y_pred_probs, axis=1) 
        
        all_y_true = np.append(all_y_true, y_test_enc)
        all_y_pred = np.append(all_y_pred, y_pred_fold)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            
        show_learning(history, title_suffix=f"- Fold {fold+1}")

    if not fold_results:
        print("\nTrening nie powiódł się w żadnym z foldów. Sprawdź komunikaty o błędach powyżej.")
        return

    print("\n--- 3. UŚREDNIONE WYNIKI K-FOLD ---")
    print(f"Wyniki dokładności dla poszczególnych foldów: {fold_results}")
    print(f"UŚREDNIONA DOKŁADNOŚĆ na {N_SPLITS} foldach: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    
    print("\n  Agregowany Raport Klasyfikacji (Precision, Recall, F1-Score):")
    target_names = class_names
    print(classification_report(all_y_true, all_y_pred, target_names=target_names, zero_division=0))
    
    print("\n--- 4. Macierz Pomyłek (Agregowana) ---")
    cm = confusion_matrix(all_y_true, all_y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Macierz Pomyłek CNN (Agregowana z K-Fold)')
    plt.ylabel('Rzeczywista Klasa')
    plt.xlabel('Przewidziana Klasa')
    plt.show() 

    if best_model:
        model_filepath = 'cnn_cgr_best_kfold_model.keras'
        best_model.save(model_filepath)
        print(f"\n Najlepszy model (dokładność: {best_accuracy:.4f}) został zapisany do pliku: {model_filepath}")




def generate_saliency_map(model, img_array, class_idx, class_names):
    """
    Generuje mapę istotności (Saliency Map) dla danego obrazu i predykcji klasy.
    Pokazuje, które piksele są najważniejsze dla decyzji modelu.
    """
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        loss = predictions[0, class_idx]

    saliency = tape.gradient(loss, img_tensor)
    
    saliency = tf.maximum(saliency, 0)
    saliency = tf.reduce_max(saliency, axis=-1) 
    saliency /= tf.reduce_max(saliency) 

    return saliency[0].numpy()

def plot_saliency_map(original_img, saliency_map, predicted_class_name, true_class_name):
    """
    Wyświetla oryginalny obraz CGR i nałożoną na niego mapę istotności.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img.squeeze(), cmap='gray')
    plt.title(f"Oryginalny CGR\nPrawdziwa klasa: {true_class_name}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(original_img.squeeze(), cmap='gray')
    plt.imshow(saliency_map, cmap='jet', alpha=0.5) 
    plt.title(f"Mapa Istotności CGR\nPrzewidziana klasa: {predicted_class_name}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

import shap

def plot_shap_explanation(shap_values, img_to_explain, class_names, predicted_class_idx):
    """
    Wizualizuje wartości SHAP, obsługując różne wersje biblioteki (formaty 4D i 5D).
    """
    print(f"Wyświetlanie wykresu SHAP dla klasy: {class_names[predicted_class_idx]}")
    
    img_reshaped = np.expand_dims(img_to_explain, axis=0)
    
    if isinstance(shap_values, list):
        to_plot = shap_values[predicted_class_idx]
    else:

        to_plot = shap_values[..., predicted_class_idx]

    shap.image_plot(to_plot, img_reshaped)

def generate_shap_explanation(model, X_background, img_to_explain):
    """
    Generuje wyjaśnienia SHAP.
    """
    explainer = shap.DeepExplainer(model, X_background)
    
    shap_values = explainer.shap_values(np.expand_dims(img_to_explain, axis=0), check_additivity=False)
    
    return shap_values



if __name__ == "__main__":
    X, y_encoded, class_names = load_data_from_images()
    
    if X is not None:
        unique, counts = np.unique(y_encoded, return_counts=True)
        print("\n" + "="*40)
        print(" DOKŁADNY SKŁAD ZBIORU DANYCH")
        print("="*40)
        for name, count in zip(class_names, counts):
            print(f" • {name:15}: {count} obrazów")
        print("-" * 40)
        print(f" RAZEM: {sum(counts)} obrazów")
        print("="*40 + "\n")
        
        train_and_evaluate_cnn_kfold(X, y_encoded, class_names)

        if os.path.exists('cnn_cgr_best_kfold_model.keras'):
            print("\n" + "="*40)
            print(" 6. ANALIZA INTERPRETOWALNOŚCI (XAI)")
            print("="*40)
            
            best_model = tf.keras.models.load_model('cnn_cgr_best_kfold_model.keras')
            best_model.trainable = True 
            
            background_indices = np.random.choice(X.shape[0], min(50, len(X)), replace=False)
            X_background = X[background_indices]

            num_samples = min(3, len(X)) 
            random_indices = np.random.choice(len(X), num_samples, replace=False)

            for i, idx in enumerate(random_indices):
                img_to_explain = X[idx]
                true_label_idx = y_encoded[idx]
                
                predictions = best_model.predict(np.expand_dims(img_to_explain, axis=0), verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                
                print(f"\n[Próbka {i+1}/{num_samples}] Index: {idx}")
                print(f"  Prawdziwa klasa: {class_names[true_label_idx]}")
                print(f"  Przewidziana: {class_names[predicted_class_idx]}")

                shap_vals = generate_shap_explanation(best_model, X_background, img_to_explain)
                plot_shap_explanation(shap_vals, img_to_explain, class_names, predicted_class_idx)
                
                saliency = generate_saliency_map(best_model, img_to_explain, predicted_class_idx, class_names)
                plot_saliency_map(img_to_explain, saliency, class_names[predicted_class_idx], class_names[true_label_idx])

            best_model.trainable = False
        else:
            print("\nBłąd: Nie znaleziono pliku modelu do analizy XAI.")

if __name__ == "__main__":
    X, y_encoded, class_names = load_data_from_images()
    
    if X is not None:

        unique, counts = np.unique(y_encoded, return_counts=True)
        print(f"\nWczytano dane. Razem: {sum(counts)} obrazów")
        

        train_and_evaluate_cnn_kfold(X, y_encoded, class_names) 

        model_path = 'cnn_cgr_best_kfold_model.keras'
        
        if os.path.exists(model_path):
            print("\n" + "="*40)
            print(" URUCHAMIANIE ANALIZY XAI Z ZAPISANEGO MODELU")
            print("="*40)

            best_model = tf.keras.models.load_model(model_path)
            best_model.trainable = True 
            
            background_indices = np.random.choice(X.shape[0], min(50, len(X)), replace=False)
            X_background = X[background_indices]

            num_samples = min(3, len(X)) 
            random_indices = np.random.choice(len(X), num_samples, replace=False)

            for i, idx in enumerate(random_indices):
                img_to_explain = X[idx]
                true_label_idx = y_encoded[idx]
                
                predictions = best_model.predict(np.expand_dims(img_to_explain, axis=0), verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                
                print(f"\n[Próbka {i+1}/{num_samples}] Index: {idx}")
                print(f"  Prawdziwa klasa: {class_names[true_label_idx]}")
                print(f"  Przewidziana: {class_names[predicted_class_idx]}")

                print("  -> Generowanie SHAP...")
                shap_vals = generate_shap_explanation(best_model, X_background, img_to_explain)
                plot_shap_explanation(shap_vals, img_to_explain, class_names, predicted_class_idx)
                
                print("  -> Generowanie Saliency Map...")
                saliency = generate_saliency_map(best_model, img_to_explain, predicted_class_idx, class_names)
                plot_saliency_map(img_to_explain, saliency, class_names[predicted_class_idx], class_names[true_label_idx])

            best_model.trainable = False
        else:
            print(f"\nBłąd: Nie znaleziono pliku {model_path}. Musisz najpierw wytrenować model chociaż raz.")


if __name__ == "__main__":
    X, y_encoded, class_names = load_data_from_images()
    
    if X is not None:
        print("\n" + "="*40)
        print(" GENEROWANIE SZCZEGÓŁOWEGO RAPORTU")
        print("="*40)
        
        import tensorflow as tf
        from sklearn.metrics import classification_report
        
        model_path = 'cnn_cgr_best_kfold_model.keras'
        
        try:
            model = tf.keras.models.load_model(model_path)
            y_pred_probs = model.predict(X)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            
          
            report = classification_report(
                y_encoded, 
                y_pred_classes, 
                target_names=class_names, 
                zero_division=0, 
                digits=6
            )
            
            print("\nSzczegółowy Raport Klasyfikacji:")
            print(report)
            
        except Exception as e:
            print(f"Błąd: {e}")

def plot_saliency_map(original_img, saliency_map, predicted_class_name, true_class_name):
    """
    Wyświetla oryginalny obraz CGR i nałożoną na niego mapę istotności (z podpisami).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(original_img.squeeze(), cmap='gray')
    axes[0].set_title(f"Oryginalna Reprezentacja CGR\n(Klasa rzeczywista: {true_class_name})", fontsize=10, pad=10)
    axes[0].axis('off')

    img1 = axes[1].imshow(original_img.squeeze(), cmap='gray')
    img2 = axes[1].imshow(saliency_map, cmap='jet', alpha=0.5)
    axes[1].set_title(f"Miejsca koncentracji filtrów sieci (Saliency Map)\n(Klasa przewidziana: {predicted_class_name})", fontsize=10, pad=10)
    axes[1].axis('off')
    
    cbar = fig.colorbar(img2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Poziom istotności cechy dla sieci (0 - niski, 1 - najwyższy)', fontsize=9)

    plt.tight_layout()

def plot_shap_explanation(shap_values, img_to_explain, class_names, predicted_class_idx):
    """
    Wizualizuje wartości SHAP i wymusza dodanie pełnych opisów osi, legendy oraz tytułu.
    Wersja bezpieczna, kompatybilna z nowym Matplotlib.
    """
    img_reshaped = np.expand_dims(img_to_explain, axis=0)
    
    if isinstance(shap_values, list):
        to_plot = shap_values[predicted_class_idx]
    else:
        to_plot = shap_values[..., predicted_class_idx]

    shap.image_plot(to_plot, img_reshaped, show=False)
    
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    
    axes = fig.get_axes()
    
    if len(axes) >= 2:
        axes[0].set_title("Oryginalny obraz DNA CGR", fontsize=10, pad=10)
        axes[0].axis('on') 
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        axes[1].set_title(f"Analiza wkładu pikseli SHAP\n(Dla predykcji klasy: {class_names[predicted_class_idx]})", fontsize=10, pad=10)
        axes[1].axis('on')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    for ax in axes:
        bbox = ax.get_position()
        if bbox.width > bbox.height and bbox.ymin < 0.25:
            ax.set_xlabel("Wartość SHAP (Czerwony: piksele potwierdzające takson | Niebieski: piksele sprzeczne)", fontsize=9, labelpad=10)
            
    plt.suptitle(f"Wyjaśnienie decyzji modelu metodą Shapleya dla klasy {class_names[predicted_class_idx]}", fontsize=12, weight='bold', y=0.98)

if __name__ == "__main__":
    X, y_encoded, class_names = load_data_from_images()
    
    if X is not None:
        model_path = 'cnn_cgr_best_kfold_model.keras'
        
        if os.path.exists(model_path):
            print("\n" + "="*50)
            print(" URUCHAMIANIE AUTOMATYCZNEJ GENERACJI XAI DLA WSZYSTKICH KLAS")
            print("="*50)
            
            print("-> Wczytywanie zapisanego modelu z dysku...")
            best_model = tf.keras.models.load_model(model_path)
            best_model.trainable = True 
            
            print("-> Przygotowywanie tła dla algorytmu SHAP (próbkowanie)...")
            np.random.seed(42) 
            background_indices = np.random.choice(X.shape[0], min(20, len(X)), replace=False)
            X_background = X[background_indices]
            
            print("-> Analizowanie bazy danych w poszukiwaniu bezbłędnych próbek...")
            all_preds_probs = best_model.predict(X, verbose=1) 
            all_preds_classes = np.argmax(all_preds_probs, axis=1)
            
            for class_idx, class_name in enumerate(class_names):
                print(f"\n" + "-"*40)
                print(f"[KLASA {class_idx+1}/8: {class_name}] Rozpoczynanie analizy...")
                
                correct_indices = np.where((y_encoded == class_idx) & (all_preds_classes == class_idx))[0]
                
                if len(correct_indices) == 0:
                    print(f"  Brak idealnej predykcji. Wybieram pierwszą próbkę z brzegu.")
                    correct_indices = np.where(y_encoded == class_idx)[0]
                
                idx = correct_indices[0]
                img_to_explain = X[idx]
                
                print(f"   • Wybrano obraz z bazy o indeksie: {idx}")

                print("   • KROK 1: Obliczanie mapy istotności (Saliency Map)...")
                saliency = generate_saliency_map(best_model, img_to_explain, class_idx, class_names)
                
                plt.figure(figsize=(11, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(img_to_explain.squeeze(), cmap='gray')
                plt.title(f"Oryginalna Reprezentacja CGR\n(Klasa rzeczywista: {class_name})", fontsize=10, pad=10)
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(img_to_explain.squeeze(), cmap='gray')
                plt.imshow(saliency, cmap='jet', alpha=0.5)
                plt.title(f"Miejsca koncentracji filtrów sieci (Saliency Map)\n(Klasa przewidziana: {class_name})", fontsize=10, pad=10)
                plt.axis('off')
                
                fig = plt.gcf()
                cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=plt.gca(), fraction=0.046, pad=0.04)
                cbar.set_label('Poziom istotności cechy dla sieci (0 - niski, 1 - najwyższy)', fontsize=9)

                plt.tight_layout()
                saliency_filename = f"saliency_{class_name}.png"
                plt.savefig(saliency_filename, dpi=150)
                plt.close()
                print(f"Zapisano plik: {saliency_filename}")

                print("   • KROK 2: Uruchamianie eksploratora SHAP (Może to potrwać dłuższą chwilę, czekaj)...")
                shap_vals = generate_shap_explanation(best_model, X_background, img_to_explain)
                
                print("   • KROK 3: Rysowanie i formatowanie wykresu SHAP...")
                plt.figure()
                plot_shap_explanation(shap_vals, img_to_explain, class_names, class_idx)
                
                shap_filename = f"shap_{class_name}.png"
                plt.savefig(shap_filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Zapisano plik: {shap_filename}")
                
            best_model.trainable = False
            print("\n" + "="*50)
            print(" PROCES ZAKOŃCZONY SUKCESEM!")

        else:
            print(f"\nBłąd: Nie znaleziono pliku {model_path} w tym katalogu.")









import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def stworz_model_cnn(input_shape=(256, 256, 3), num_classes=8):
    """
    Funkcja budująca strukturę głębokiej sieci splotowej (CNN)
    """
    model = Sequential([

        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),               
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    print("Inicjalizacja i budowanie modelu CNN...")
    model = stworz_model_cnn(input_shape=(256, 256, 3), num_classes=8)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    nazwa_pliku = 'model_architektura.keras'
    model.save(nazwa_pliku)
