import os
import time
from colorama import Fore, Style, init
from libs.ExecuteModel import single_image, multi_image

# Inisialisasi colorama untuk Windows
init(autoreset=True)

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} detik"
    else:
        menit = int(seconds // 60)
        detik = int(seconds % 60)
        return f"{menit} menit {detik} detik"

def TextToImage(Model, Prompt, Image, single=True):
    hasil = ""

    if single:
        print(Fore.YELLOW + "Model Selected:", Model)
        print(Fore.YELLOW + "Prompt Selected:", Prompt)
        print(Fore.YELLOW + "Image Selected:", Image)
        hasil = single_image(Model, Prompt, Image)
        return hasil

    else:
        image_files = [f for f in os.listdir(Image) if f.lower().endswith(('.jpg', '.png'))]
        total = len(image_files)
        start_time = time.time()

        for i, filename in enumerate(image_files, start=1):
            full_path = os.path.join(Image, filename)
            # Hanya menampilkan garis pendek sekali di sini
            print(Fore.LIGHTBLACK_EX + "-"*100)  # Garis pendek di awal pertama

            print(Fore.CYAN + f"[{i} / {total}] Processing: {full_path}")
            hasil += f"[{i} / {total}] Image Selected: {full_path}\n"

            iter_start = time.time()

            if os.path.exists(full_path):
                result = multi_image(Model, Prompt, full_path)
                hasil += result + "\n"
                print(Fore.LIGHTBLACK_EX + "=-"*16)
                print(Fore.GREEN + "✅ Success")
            else:
                msg = f"❌ File tidak ditemukan: {full_path}"
                hasil += msg + "\n"
                print(Fore.RED + msg)

            # Hitung waktu
            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            remaining = (elapsed / i) * (total - i) if i != 0 else 0

            # Log ke terminal
            print(Fore.BLUE + f"⏱ Waktu proses gambar : {format_time(iter_time)}")
            print(Fore.MAGENTA + f"⏳ Total waktu berlalu : {format_time(elapsed)}")
            print(Fore.YELLOW + f"🕒 Estimasi sisa waktu  : {format_time(remaining)}")

            # Tambahkan ke hasil output teks
            hasil += (
                f"⏱ Waktu proses gambar : {format_time(iter_time)}\n"
                f"⏳ Total waktu berlalu : {format_time(elapsed)}\n"
                f"🕒 Estimasi sisa waktu : {format_time(remaining)}\n"
                "--------------------------------------------------\n"
            )

        return hasil
