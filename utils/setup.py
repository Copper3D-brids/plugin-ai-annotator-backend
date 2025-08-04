from pathlib import Path
from dotenv import load_dotenv
import os
import sys
from huggingface_hub import snapshot_download  # Install huggingface_hub if not already installed


# --- Download Trained Model Weights (~400MB) ---
def download_nn_interactive(repo_id, model_name, train_dataset_folder):
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{model_name}/*"],
        local_dir=train_dataset_folder
    )


def get_base_from_env():
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)

    if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        return os.environ["BASE"]
    elif sys.platform.startswith('win'):
        return os.environ["BASE_DUKE_locally"]
        # return os.environ["BASE_locally"]
    return os.environ["BASE"]


class Config:
    MASKS = None
    METADATA = None
    Connected_Websocket = None
    Updated_Mesh = False
    ClearAllMask = False
    Current_Case_Name = ""
    BASE_PATH = Path(get_base_from_env())
    METADATA_PATH = "./manifest.xlsx"
    MASK_FILE_PATH = ""
    MASK_FOLDER_PATH = ""
    IMPORT_FOLDER_PATH = "import_nrrd"
    EXPORT_FOLDER_PATH = "export_data"
    CASE_NAMES = []
    NNInteractive_REPO_ID = "nnInteractive/nnInteractive"
    NNInteractive_MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
    NNInteractive_TRAIN_DIR = "./nninteractive"


class TumourData:
    volume: 0
    extent: 0
    skin: 0
    ribcage: 0
    nipple: 0
