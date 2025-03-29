from .data.download import download_dataset, extract_files
from .plotting.fonts import load_font, FONTS_URLS
from .plotting.formatting import make_fig_pretty, PATTERNS
from .ml.scaling import CustomScaler, resize_images_in_folder
from .ml.training import train_model
from .ml.metrics import r2_score, accuracy_score
from .__version__ import __version__

__all__ = [
    'download_dataset',
    'extract_files',
    'load_font',
    'FONTS_URLS',
    'make_fig_pretty',
    'PATTERNS',
    'CustomScaler',
    'resize_images_in_folder',
    'train_model',
    'r2_score',
    'accuracy_score',
    '__version__'
]


print('Faculty of Science and Engineering 🔬')
print('\033[95mThe University of Manchester \033[0m')
print(f'Invoking utils version: \033[92m{__version__}\033[0m')
