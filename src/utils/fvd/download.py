import requests
from tqdm import tqdm
import os
import torch


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 8192

    pbar = tqdm(total=0, unit="iB", unit_scale=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()


def download(id, fname, root=os.path.expanduser("~/.cache/videogpt")):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)
    return destination


_I3D_PRETRAINED_ID = "1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT"


def load_i3d_pretrained(device=torch.device("cpu")):
    from src.utils.fvd.pytorch_i3d import InceptionI3d

    i3d = InceptionI3d(400, in_channels=3).to(device)
    filepath = download(_I3D_PRETRAINED_ID, "i3d_pretrained_400.pt")
    # filepath = "./models/i3d/i3d_pretrained_400.pt"
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d
