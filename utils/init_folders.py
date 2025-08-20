import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
FOLDERS = ['videos','subtitles','downloads','models','outputs','logs']
def ensure():
    for d in FOLDERS:
        path = os.path.join(BASE_DIR, d)
        os.makedirs(path, exist_ok=True)
def get_path(folder, name):
    return os.path.join(os.path.abspath(os.path.dirname(__file__) + "/.."), folder, name)
