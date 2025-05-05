import os
import sys

def list_prompts():
    folder = "./prompt"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return [f for f in os.listdir(folder) if f.endswith(".txt")]

def list_models():
    return ["unsloth/llama-3-8b-Instruct", "commonpool_xl_clip_s13b_b90k", "ViT-B-32"]

def get_file_content(filename):
    try:
        with open(os.path.join("prompt", filename), 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "Gagal membuka file."

def save_file_content(filename, content):
    try:
        with open(os.path.join("prompt", filename), 'w', encoding='utf-8') as f:
            f.write(content)
        return "Berhasil disimpan."
    except:
        return "Gagal menyimpan file."

def delete_file(filename):
    try:
        os.remove(os.path.join("prompt", filename))
        return "Berhasil dihapus."
    except:
        return "Gagal menghapus file."

def create_new_file(filename):
    if not filename.endswith(".txt"):
        return "Gunakan ekstensi .txt"
    path = os.path.join("prompt", filename)
    if os.path.exists(path):
        return "File sudah ada."
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("")
        return "File berhasil dibuat."
    except:
        return "Gagal membuat file."

def restart_python():
    os.execv(sys.executable, ['python'] + sys.argv)

def generate_captions_bulk(folder_path, prompt, model):
    if not os.path.isdir(folder_path):
        return "Folder tidak ditemukan."
    return f"[{model}] [{prompt}] Caption untuk semua gambar di folder '{folder_path}' telah dihasilkan."

def baca_file(nama_file):
    nama_file = "./prompt/" + nama_file
    try:
        with open(nama_file, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return f"File '{nama_file}' tidak ditemukan."
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"