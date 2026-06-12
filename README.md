# Interpretowalna analiza filogenetyczna mitochondrialnego DNA z wykorzystaniem metod uczenia maszynowego i wizualizacji CGR

Repozytorium zawiera kompletny potok programistyczny (*pipeline*) opracowany w ramach pracy magisterskiej, służący do wielkoskalowego pobierania, mutowania, reprezentacji i klasyfikacji sekwencji genomicznych.

## 1. Opis projektu
System opiera się na transformacji liniowych sekwencji nukleotydowych (DNA) do dwuwymiarowych obrazów gęstości za pomocą metody **Chaos Game Representation (CGR)**. Wygenerowane w ten sposób reprezentacje geometryczne stanowią wejście dla głębokiej sieci splotowej (**CNN**), która dokonuje automatycznej klasyfikacji taksonomicznej obiektów biologicznych. 

Projekt zawiera również moduł augmentacji danych realizujący symulacje ewolucyjne (mutacje synonimiczne, niesynonimiczne oraz w regionach kontrolnych D-loop) na podstawie plików adnotacji GFF.

---

## 2. Architektura potoku przetwarzania (Struktura plików)

Struktura repozytorium odpowiada kolejnym etapom przetwarzania i analizy danych:

* `download_fasta_batch.py` – Skrypt do masowego, automatycznego pobierania surowych sekwencji genomicznych w formacie FASTA z bazy danych NCBI Nuccore (E-utilities) na podstawie listy identyfikatorów (*accession numbers*).
* `download_gff.py` – Skrypt realizujący pobieranie powiązanych plików adnotacji strukturalnych genomu (GFF3) z bazy NCBI, niezbędnych do identyfikacji regionów kodujących oraz kontrolnych.
* `warianty.py` – Algorytm ewolucyjnej augmentacji danych. Na podstawie sekwencji wejściowych oraz cech z plików GFF generuje zmutowane warianty syntetyczne z zachowaniem biologicznego rygoru (symulacja mutacji synonimicznych, niesynonimicznych i hiperzmiennych w regionie D-loop).
* `generuj.py` – Moduł generowania obrazów CGR. Przekształca łańcuchy nukleotydów na współrzędne dwuwymiarowe $X, Y$ oraz eksportuje wyniki do postaci plików graficznych PNG o wysokiej rozdzielczości oraz powiązanych zestawów danych.
* `uczenie.py` – Główny moduł inżynierii uczenia maszynowego. Odpowiada za wczytanie danych, podział w ramach $K$-krotnej walidacji krzyżowej (*K-Fold Cross-Validation*), definicję struktury głębokiej sieci splotowej (CNN), trening, walidację oraz ewaluację metryk skuteczności klasyfikatora.
* `predykcja.py` – Skrypt predykcyjny (inferencyjny), umożliwiający załadowanie wytrenowanego modelu w formacie `.keras` i przeprowadzenie automatycznego rozpoznania przynależności taksonomicznej dla nowej, niezależnej próbki obrazu CGR.

---

## 3. Wymagania systemowe i instalacja

Projekt został zaimplementowany w języku Python 3.11 z użyciem frameworka TensorFlow/Keras.
