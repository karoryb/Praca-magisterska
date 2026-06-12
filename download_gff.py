import os
import requests
import time
from tqdm import tqdm


EMAIL = "288847@student.pwr.edu.pl"
API_KEY = None 
MAX_TRIES = 5
WAIT_BETWEEN = 0.34 
INPUT_FILE = "accessions.txt"
OUTPUT_DIR = "GFF" 

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_gff(acc):
    """Pobiera plik adnotacji GFF3 dla danego accession z NCBI E-utilities."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "nuccore",
        "id": acc,
        "rettype": "gff3", 
        "retmode": "text",
        "email": EMAIL,
    }
    if API_KEY:
        params["api_key"] = API_KEY

    headers = {"User-Agent": f"gff-downloader/1.0 ({EMAIL})"}

    for attempt in range(1, MAX_TRIES + 1):
        try:

            r = requests.get(base_url, params=params, headers=headers, timeout=30) 

            if r.status_code == 200 and len(r.text) > 50:

                 if r.text.startswith("##gff-version 3") or r.text.startswith("#"): 
                    return r.text
                 else:
                    print(f"[{acc}] Serwer zwrócił nieoczekiwaną treść. Sprawdzanie czy accession jest poprawne.")
                    r.raise_for_status() 
                    
            elif r.status_code in (429, 500, 502, 503, 504):
                sleep_time = 2 ** (attempt - 1) + random.uniform(0, 1)
                print(f"[{acc}] Ograniczenie lub błąd serwera ({r.status_code}), ponawiam za {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            else:
                r.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            sleep_time = 2 ** (attempt - 1) + random.uniform(0, 1)
            print(f"[{acc}] Błąd połączenia lub HTTP: {e}, ponawiam próbę za {sleep_time:.2f}s...")
            time.sleep(sleep_time)

    print(f"[{acc}] Nie udało się pobrać GFF po {MAX_TRIES} próbach.")
    return None


def main():
    """Główna funkcja wczytująca accessions i pobierająca pliki GFF."""
    try:
        with open(INPUT_FILE, 'r') as f:
            accessions = [line.strip().split('.')[0] for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku wejściowego '{INPUT_FILE}'. Upewnij się, że istnieje.")
        return

    print(f"Znaleziono {len(accessions)} identyfikatorów do pobrania.\n")

    for acc in tqdm(accessions, desc="Pobieranie GFF"):

        gff_data = fetch_gff(acc)
        
        if gff_data:

            output_path = os.path.join(OUTPUT_DIR, f"{acc}.gff3")
            
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(gff_data)
                
            except Exception as e:
                print(f"[{acc}] Błąd zapisu pliku GFF: {e}")
                
        time.sleep(WAIT_BETWEEN) 

    print(f"\nZapisano pliki GFF/GFF3 w folderze: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()