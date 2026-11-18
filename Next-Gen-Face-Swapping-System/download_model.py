import os
import sys
import urllib.request
import bz2

MODEL_URL = "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
BZ2_PATH = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat.bz2")
DAT_PATH = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
    else:
        sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f} MB")
    sys.stdout.flush()

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(DAT_PATH) and os.path.getsize(DAT_PATH) > 90 * 1024 * 1024:
        print(f"Model already exists at {DAT_PATH}")
        return

    print(f"Downloading landmark predictor model from {MODEL_URL}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(MODEL_URL, headers=headers)
    with urllib.request.urlopen(req) as response, open(BZ2_PATH, 'wb') as out_file:
        total_size = int(response.headers.get('Content-Length', 0))
        block_size = 1024 * 1024
        downloaded = 0
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            reporthook(downloaded // block_size, block_size, total_size)
    print("\nDownload complete. Decompressing bz2 archive...")

    with bz2.BZ2File(BZ2_PATH, 'rb') as f_in, open(DAT_PATH, 'wb') as f_out:
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)

    print(f"Extraction complete: {DAT_PATH} ({os.path.getsize(DAT_PATH) / (1024*1024):.1f} MB)")
    if os.path.exists(BZ2_PATH):
        os.remove(BZ2_PATH)
        print("Removed temporary compressed file.")

if __name__ == "__main__":
    main()
