from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'default_image.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'How to Get a Dent Out of a Bumper.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'video_1.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'LifeHacks - Using Boiling Water to Get Car Dents Out.mp4'
VIDEO_4_PATH = VIDEO_DIR / 'VIDEO_4.mp4'
VIDEO_5_PATH = VIDEO_DIR / 'VIDEO_5.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
    'video_4': VIDEO_4_PATH,
    'video_5': VIDEO_5_PATH,
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETERIORATION_MODEL = MODEL_DIR / 'road_best.pt'
ROADFURNITURE_MODEL = MODEL_DIR / 'traffic_best.pt'


# Webcam
WEBCAM_PATH = 0
