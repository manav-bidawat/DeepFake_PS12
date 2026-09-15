import os
import argparse
import re
import csv
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path

# Standard import for SpeechBrain inference
from speechbrain.inference.speaker import EncoderClassifier

def parse_real_folder(foldername):
    """Strictly matches 'speaker_X' and ignores any noise folders."""
    match = re.search(r'^speaker_(\d+)$', foldername, re.IGNORECASE)
    return int(match.group(1)) if match else None

def parse_sync_folder(foldername):
    """Strictly matches 'speakerX' and ignores any noise folders."""
    match = re.search(r'^speaker(\d+)$', foldername, re.IGNORECASE)
    return int(match.group(1)) if match else None

def get_voice_id_real(filename):
    """Extracts the integer ID from real filenames like '1.wav'."""
    match = re.search(r'^0*(\d+)\.(wav|flac|ogg)$', filename, re.IGNORECASE)
    return int(match.group(1)) if match else None

def get_voice_id_sync(filename):
    """Extracts the integer ID from synthetic filenames like 'spk1_03.wav'."""
    # Changed from ^speaker\d+ to ^spk\d+ to match your new naming convention
    match = re.search(r'^spk\d+_0*(\d+)\.(wav|flac|ogg)$', filename, re.IGNORECASE)
    return int(match.group(1)) if match else None

def build_file_map(folder_path, id_extractor):
    """Creates a dictionary mapping voice ID to its full file path."""
    file_map = {}
    path = Path(folder_path)
    if not path.is_dir():
        return file_map

    for f in path.iterdir():
        if f.is_file():
            voice_id = id_extractor(f.name)
            if voice_id is not None:
                file_map[voice_id] = str(f)
    return file_map

def extract_embedding(audio_path, classifier):
    """Loads audio, ensures 16kHz mono, and extracts ECAPA-TDNN embedding."""
    signal, fs = torchaudio.load(audio_path)

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)

    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        return embeddings.squeeze()

def main():
    parser = argparse.ArgumentParser(description="Evaluate clean audio directories using ECAPA-TDNN")
    parser.add_argument("--real", type=str, default="data/audio/real", help="Base directory for real audio")
    parser.add_argument("--sync", type=str, default="data/audio/synthetic", help="Base directory for synthetic audio")
    parser.add_argument("--output", type=str, default="clean_similarity_results.csv", help="Output CSV file")

    args = parser.parse_args()

    print("Loading ECAPA-TDNN model from SpeechBrain...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmp_ecapa_model"
    )

    real_base = Path(args.real)
    sync_base = Path(args.sync)

    if not real_base.is_dir() or not sync_base.is_dir():
        print("Error: The specified base directories do not exist.")
        return

    # 1. Map the folders dynamically based strictly on Speaker Number
    print("\nMapping directory structures (ignoring noise folders)...")
    folder_pairs = {} # key: spk_num, value: {'real': path, 'sync': path}

    for real_dir in real_base.iterdir():
        if real_dir.is_dir():
            spk_num = parse_real_folder(real_dir.name)
            if spk_num is not None:
                folder_pairs[spk_num] = {'real': real_dir}

    for sync_dir in sync_base.iterdir():
        if sync_dir.is_dir():
            spk_num = parse_sync_folder(sync_dir.name)
            if spk_num is not None and spk_num in folder_pairs:
                folder_pairs[spk_num]['sync'] = sync_dir

    # Filter out unmatched pairs
    matched_pairs = {k: v for k, v in folder_pairs.items() if 'real' in v and 'sync' in v}
    print(f"Successfully matched {len(matched_pairs)} clean speaker folders.")

    results = []
    # Dictionary to store scores grouped by speaker
    speaker_scores = {}

    print("\nProcessing Audio Files...")

    for spk_num, paths in sorted(matched_pairs.items()):
        print(f"  -> Comparing Speaker {spk_num}")

        real_files = build_file_map(paths['real'], get_voice_id_real)
        sync_files = build_file_map(paths['sync'], get_voice_id_sync)

        speaker_name = f"Speaker {spk_num}"
        speaker_scores[speaker_name] = []

        # Cache embeddings to save redundant computation
        real_embeddings_cache = {}

        for voice_id, sync_path in sync_files.items():
            if voice_id in real_files:
                real_path = real_files[voice_id]

                if voice_id not in real_embeddings_cache:
                    real_embeddings_cache[voice_id] = extract_embedding(real_path, classifier)
                emb_real = real_embeddings_cache[voice_id]

                emb_sync = extract_embedding(sync_path, classifier)
                similarity = F.cosine_similarity(emb_real.unsqueeze(0), emb_sync.unsqueeze(0)).item()

                speaker_scores[speaker_name].append(similarity)

                results.append({
                    "Speaker": speaker_name,
                    "Voice_ID": voice_id,
                    "Real_File": Path(real_path).name,
                    "Synthetic_File": Path(sync_path).name,
                    "Cosine_Similarity": round(similarity, 4)
                })

    # --- Save CSV ---
    if results:
        with open(args.output, mode='w', newline='') as csv_file:
            fieldnames = ['Speaker', 'Voice_ID', 'Real_File', 'Synthetic_File', 'Cosine_Similarity']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\n[+] Data saved to: {args.output}")
    else:
        print("\n[-] No matching files found to compare. Please check file naming conventions.")
        return

    # --- Print Reporting ---
    print("\n" + "="*45)
    print("ECAPA-TDNN SIMILARITY REPORT (CLEAN AUDIO)")
    print("="*45)

    all_scores = []

    for speaker_name in sorted(speaker_scores.keys()):
        scores = speaker_scores[speaker_name]
        if scores:
            avg_score = sum(scores) / len(scores)
            all_scores.extend(scores)
            print(f"{speaker_name} Average Score: {avg_score:.4f}  (Based on {len(scores)} files)")

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
    print("-" * 45)
    print(f"TOTAL OVERALL AVERAGE:    {overall_avg:.4f}  (Based on {len(all_scores)} files)")
    print("="*45)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
