import wfdb
import os
from collections import Counter

def check_all_patients():
    data_dir = 'mitdb'
    
    # 1. Get all Patient IDs from the files in the directory
    # We look for .dat files and strip the extension to get the ID
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found. Please run download_data.py first.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    patient_ids = sorted([f.split('.')[0] for f in files])

    print(f"{'Patient ID':<12} | {'Normal (N)':<12} | {'Abnormal (V)':<12} | {'Paced (/)':<12} | {'Status'}")
    print("-" * 70)

    good_patients = []
    bad_patients = []

    for pid in patient_ids:
        try:
            # Read annotations
            annotation = wfdb.rdann(os.path.join(data_dir, pid), 'atr')
            counts = Counter(annotation.symbol)
            
            n_count = counts.get('N', 0)
            v_count = counts.get('V', 0)
            p_count = counts.get('/', 0)
            
            # Determine Status
            if n_count > 1000:
                status = "GOOD"
                good_patients.append(pid)
            elif n_count == 0:
                status = "BAD (No N)"
                bad_patients.append(pid)
            else:
                status = "MIXED"
                # We might still use mixed patients if they have enough N beats
                if n_count > 500: 
                    good_patients.append(pid)
                else:
                    bad_patients.append(pid)

            print(f"{pid:<12} | {n_count:<12} | {v_count:<12} | {p_count:<12} | {status}")

        except Exception as e:
            print(f"{pid:<12} | ERROR READING FILE")

    print("-" * 70)
    print(f"\nRecommended 'Clean List' for task.py ({len(good_patients)} patients):")
    print(good_patients)
    
    print(f"\nPatients to Exclude ({len(bad_patients)} patients):")
    print(bad_patients)

if __name__ == "__main__":
    check_all_patients()
