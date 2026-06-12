import os
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
import pandas as pd

def generuj_cgr_i_wspolrzedne(sekwencja):
    """
    Generuje punkty CGR i zwraca ich współrzędne wraz z indeksem nukleotydu.
    Zwracany format: [x, y, indeks_nukleotydu]
    """
    n = len(sekwencja)
    punkty_z_indeksami = []
    poprzedni_punkt = np.array([0.5, 0.5]) 


    for i in range(n):
        nukleotyd = sekwencja[i].upper()
        
        if nukleotyd == 'A':
            rog = np.array([0, 0])
        elif nukleotyd == 'C':
            rog = np.array([0, 1])
        elif nukleotyd == 'G':
            rog = np.array([1, 1])
        elif nukleotyd == 'T':
            rog = np.array([1, 0])
        else:
            continue 

        aktualny_punkt = 0.5 * (poprzedni_punkt + rog)

        punkty_z_indeksami.append(np.array([aktualny_punkt[0], aktualny_punkt[1], i + 1])) 
        
        poprzedni_punkt = aktualny_punkt

    return np.array(punkty_z_indeksami)


folder_wejscie = r"C:\Users\Karola\OneDrive - MULTIKKA Sp. z o.o\Dokumenty\STUDIA\INS\magisterka\ZMUTOWANE_SEKWENCJE"
folder_wyjscie = r"C:\Users\Karola\OneDrive - MULTIKKA Sp. z o.o\Dokumenty\STUDIA\INS\magisterka\CGRpng"
folder_wspolrzedne = r"C:\Users\Karola\OneDrive - MULTIKKA Sp. z o.o\Dokumenty\STUDIA\INS\magisterka\CGR_Coordinates"

os.makedirs(folder_wyjscie, exist_ok=True)
os.makedirs(folder_wspolrzedne, exist_ok=True)


for plik in os.listdir(folder_wejscie):
    if plik.lower().endswith(".fasta"):
        sciezka = os.path.join(folder_wejscie, plik)
        
        for rekord in SeqIO.parse(sciezka, "fasta"):
            sekwencja = str(rekord.seq)
            
            dane_cgr = generuj_cgr_i_wspolrzedne(sekwencja)

            punkty_xy = dane_cgr[:, :2].T 

            plt.figure(figsize=(5,5))
            plt.plot(punkty_xy[0, :], punkty_xy[1, :], '.', markersize=1, color='black')
            plt.axis('off')
            plt.gca().set_position([0, 0, 1, 1])

            nazwa_podstawowa = os.path.splitext(plik)[0]
            nazwa_png = f"{nazwa_podstawowa}.png"
            sciezka_png = os.path.join(folder_wyjscie, nazwa_png)
            plt.savefig(sciezka_png, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            df_wsp = pd.DataFrame({
                'Index_Nukleotydu': dane_cgr[:, 2].astype(int),
                'X': dane_cgr[:, 0],
                'Y': dane_cgr[:, 1]
            })
            
            nazwa_csv = f"{nazwa_podstawowa}.csv"
            sciezka_csv = os.path.join(folder_wspolrzedne, nazwa_csv)
            df_wsp.to_csv(sciezka_csv, index=False)
            
            print(f"Przetworzono {plik}: zapisano PNG i CSV ze współrzędnymi.")

print("\n--- Zakończenie Przetwarzania ---")
print(f"Obrazy CGR zapisano w: {folder_wyjscie}")
print(f"Współrzędne CGR zapisano w: {folder_wspolrzedne}")