import os
import random
from Bio import SeqIO, Seq
from Bio.Data import CodonTable
import pandas as pd


folder_wejsciowy_fasta = r"C:\Users\Karola\OneDrive - MULTIKKA Sp. z o.o\Dokumenty\STUDIA\INS\magisterka\fasta"
folder_wejsciowy_gff = r"C:\Users\Karola\OneDrive - MULTIKKA Sp. z o.o\Dokumenty\STUDIA\INS\magisterka\GFF"
folder_wyjsciowy_fasta = r"C:\Users\Karola\OneDrive - MULTIKKA Sp. z o.o\Dokumenty\STUDIA\INS\magisterka\ZMUTOWANE_SEKWENCJE"

os.makedirs(folder_wyjsciowy_fasta, exist_ok=True)

NUM_VARIANTS = 10 

MITO_TABLE = CodonTable.unambiguous_dna_by_id[2]
NUCLEOTIDES = ['A', 'T', 'C', 'G']


PROB_CR = 0.005                                      # Region Kontrolny/D-loop (wysoka zmienność)
PROB_SYN = 0.001                                     # Mutacja Synonimiczna na kodon (średnia)
PROB_NONSYN = 0.0001                                 # Mutacja Niesynonimiczna na kodon (niska)
PROB_tRNA_rRNA = 0.00005                             # tRNA/rRNA (bardzo niska)
PROB_SNV_OTHER = 0.0005                              # Inne regiony nieadnotowane (umiarkowana)


TRANSITION_BIAS = 0.8 
TRANSITIONS = {'A': 'G', 'G': 'A', 'C': 'T', 'T': 'C'}
TRANSVERSIONS = {'A': ['C', 'T'], 'G': ['C', 'T'], 'C': ['A', 'G'], 'T': ['A', 'G']}

def choose_new_nucleotide(original_nt):
    """Wybiera nowy nukleotyd z uwzględnieniem biasu przejścia/transwersji."""
    if random.random() < TRANSITION_BIAS:
        # translacja 
        return TRANSITIONS.get(original_nt, random.choice([nt for nt in NUCLEOTIDES if nt != original_nt]))
    else:
        # transwersja
        return random.choice(TRANSVERSIONS.get(original_nt, [nt for nt in NUCLEOTIDES if nt != original_nt]))

def get_synonymous_change(codon):
    """Znajduje losową synonimiczną zmianę dla kodonu, preferując 1 SNV."""
    current_aa = MITO_TABLE.forward_table.get(codon)
    if not current_aa:
        return None

    synonyms = []
    for c, aa in MITO_TABLE.forward_table.items():
        if aa == current_aa and c != codon:
            synonyms.append(c)

    one_snv_synonyms = [s for s in synonyms if sum(s[i] != codon[i] for i in range(3)) == 1]
    
    if one_snv_synonyms:
        return random.choice(one_snv_synonyms)
    elif synonyms:
        return random.choice(synonyms)
    else:
        return None

def parse_gff(gff_file_path):
    """Odczytuje tylko najważniejsze cechy (typy) z pliku GFF za pomocą Pandas."""
    try:
        df = pd.read_csv(gff_file_path, sep='\t', comment='#', header=None, 
                         names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'])
    except Exception as e:
        print(f"Błąd podczas wczytywania GFF {gff_file_path}: {e}")
        return []

    features = []

    relevant_features = ['gene', 'rRNA', 'tRNA', 'D-loop', 'Control region']
    
    for index, row in df.iterrows():
        feature_type = row['feature']
        if feature_type in relevant_features:
            features.append({
                'type': feature_type,
                'start': row['start'] - 1, 
                'end': row['end'], 
                'strand': row['strand']
            })
    return features



def generate_variants(original_seq, features, num_variants):
    """Generuje zmutowane warianty na podstawie sekwencji i adnotacji."""
    variants = []
    seq_len = len(original_seq)
    
    for v_index in range(num_variants):
        mutated_seq_list = list(str(original_seq))

        for pos in range(seq_len):

            is_annotated = False

            for feature in features:
                if feature['start'] <= pos < feature['end']:
                    is_annotated = True
                    break
            
            if not is_annotated:
                 if random.random() < PROB_SNV_OTHER:
                    mutated_seq_list[pos] = choose_new_nucleotide(original_seq[pos])


        for feature in features:
            f_start = feature['start'] 
            f_end = feature['end'] 
            f_type = feature['type']
            f_strand = feature['strand']

            if f_type in ['D-loop', 'Control region', 'rRNA', 'tRNA']:
                prob = PROB_CR if f_type in ['D-loop', 'Control region'] else PROB_tRNA_rRNA
                for pos in range(f_start, f_end):
                    if random.random() < prob:
                        mutated_seq_list[pos] = choose_new_nucleotide(original_seq[pos])
            
            elif f_type == 'gene':
                for codon_start in range(f_start, f_end, 3):
                    codon_end = codon_start + 3
                    if codon_end > f_end:
                        continue 

                    original_codon = "".join(original_seq[codon_start:codon_end])

                    if random.random() < PROB_SYN:
                        if f_strand == '-':
                            temp_codon = str(Seq.Seq(original_codon).reverse_complement())
                        else:
                            temp_codon = original_codon
                            
                        new_codon = get_synonymous_change(temp_codon)
                        
                        if new_codon:
                            if f_strand == '-':
                                final_codon_change = str(Seq.Seq(new_codon).reverse_complement())
                            else:
                                final_codon_change = new_codon

                            for i in range(3):
                                mutated_seq_list[codon_start + i] = final_codon_change[i]
                            continue 

                    if random.random() < PROB_NONSYN:
                        nt_to_change_index = random.randint(0, 2) 
                        change_pos = codon_start + nt_to_change_index
                        
                        original_nt = mutated_seq_list[change_pos]
                        mutated_seq_list[change_pos] = choose_new_nucleotide(original_nt)

        variants.append(Seq.Seq("".join(mutated_seq_list)))

    return variants


def main():
    """Główna funkcja przetwarzająca pliki z dwóch różnych folderów."""
    print(f"--- START PRZETWARZANIA ({NUM_VARIANTS} wariantów na gatunek) ---")

    fasta_files = [f for f in os.listdir(folder_wejsciowy_fasta) if f.endswith('.fasta') or f.endswith('.fa')]
    
    for fasta_filename in fasta_files:
        base_name_full = os.path.splitext(fasta_filename)[0]

        base_name_no_version = base_name_full.split('.')[0] 
        
        fasta_path = os.path.join(folder_wejsciowy_fasta, fasta_filename)
        gff_filename_gff3 = base_name_no_version + '.gff3'
        gff_path = os.path.join(folder_wejsciowy_gff, gff_filename_gff3)
        
        if not os.path.exists(gff_path):
            gff_filename_gff = base_name_no_version + '.gff'
            gff_path = os.path.join(folder_wejsciowy_gff, gff_filename_gff)
            
            if not os.path.exists(gff_path):
                print(f" Nie znaleziono pliku GFF dla {base_name_full} (szukano {gff_filename_gff3} lub {gff_filename_gff}). Pomijam.")
                continue
            
        print(f"\nPrzetwarzanie: {base_name_full} (znaleziono GFF: {os.path.basename(gff_path)})")

        try:
            record = SeqIO.read(fasta_path, "fasta")
            original_seq = record.seq.upper()
        except Exception as e:
            print(f"Błąd wczytywania FASTA: {e}")
            continue

        features = parse_gff(gff_path)
        if not features:
            print(f"Nie udało się wczytać cech z GFF. Pomijam.")
            continue 

        mutated_variants = generate_variants(original_seq, features, NUM_VARIANTS)
        
        save_count = 0
        for i, seq in enumerate(mutated_variants):
            new_id = f"{record.id}_VAR{i+1}"
            new_description = f"Mutowany wariant {i+1} z {record.id}"
            output_record = SeqIO.SeqRecord(seq, id=new_id, description=new_description)
            
            output_fasta_path = os.path.join(folder_wyjsciowy_fasta, f"{base_name_full}_VAR{i+1}.fasta")
            
            try:
                with open(output_fasta_path, "w") as output_handle:
                    SeqIO.write(output_record, output_handle, "fasta")
                save_count += 1
            except Exception as e:
                print(f"Błąd zapisu pliku {output_fasta_path}: {e}")
                
        print(f"Pomyślnie zapisano {save_count} osobnych wariantów dla {base_name_full}.")



if __name__ == "__main__":
    main()