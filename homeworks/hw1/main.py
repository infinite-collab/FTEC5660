from pathlib import Path
from google import genai
from google.genai import types
# You should put all photos in ./photos directory
PHOTO_DIR = "./photos"
MODEL = "gemini-3-pro-preview"
BASE_URL = input("Enter the API base URL (default: https://api.zetatechs.com),the official google API url should be https://generativelanguage.googleapis.com: ") or "https://api.zetatechs.com"
api_key = input("Enter your API key: ")
def load_photos(directory):
    directory = Path(directory)
    parts = []
    files = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".jpg",".jpeg"]:
            files.append(p)
    files.sort()
    for p in files:
        img_bites = p.read_bytes()
        parts.append(types.Part.from_bytes(data=img_bites, mime_type="image/jpeg"))
    return parts,files

def main():
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            base_url=BASE_URL,
            api_version="v1beta",
        ),
    )
    img_parts, files = load_photos(PHOTO_DIR)
    print(f"Loaded {len(img_parts)} JPEG images from {PHOTO_DIR}:")
    if not img_parts:
        raise ValueError("No JPEG images found in the specified directory.")
    chat = client.chats.create(
        model = MODEL,
        config=types.GenerateContentConfig(
            system_instruction="Now you receive multiple billing recipes,anwer the user query, and reject irrelevant queries",
                    thinking_config=types.ThinkingConfig(
            thinking_level="high",   
            include_thoughts=True    
        ),
        temperature=0.2,
    )
    )

    resp = chat.send_message(["this is recipe photos",*img_parts])
    while True:
        q = input("Enter your question about the photos (or 'exit' to quit): ")
        if q.lower() == "exit":
            break
        resp = chat.send_message(q)
        print("Response:", resp.text)

if __name__ == "__main__":
    main()
