from pathlib import Path
from tqdm import tqdm
import requests
import tarfile
import zipfile
import json

__all__ = ['download_dataset', 'extract_files']

def find_project_root() -> Path:
    current_path = Path(__file__).resolve()
    while current_path != current_path.root:
        if (current_path / 'utils').exists():  # Check if 'utils' directory exists
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Project root not found")

def download_dataset(dataset_name:str, dest_path: str, extract: bool = False, remove_compressed: bool = False) -> Path:
    """"
    Download a dataset from a URL and extract it to a specified path"
    Args:
        url (str): URL of the dataset
        dest_path (str): Path to save the dataset
        extract (bool): Extract the dataset if it is compressed
        remove_compressed (bool): Remove the compressed file after extraction
    Returns:
        Path: Path to the extracted dataset
    """
    # Load the dataset URL from a JSON file
    project_root = find_project_root()
    json_path = project_root / 'utils/data/datasets.json'
    # json_path = Path('utils/data/datasets.json')
    if not json_path.exists():
        print('ERROR: datasets.json file not found')
        return None
   
    with open(json_path, 'r', encoding='utf-8') as f:
        datasets = json.load(f)
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not found in datasets.json")
    
    url = datasets[dataset_name]['url']
    desc = datasets[dataset_name]['description']
    authors = ", ".join(datasets[dataset_name]['authors'])
    year = datasets[dataset_name]['year']
    website = datasets[dataset_name]['website']

    print(f'Downloading:\n{desc}')
    print(f'> Authors: {authors}')
    print(f'> Year: {year}')
    print(f'> Website: {website}\n')

    dest_path = Path(dest_path) if dest_path else Path.cwd()
    f_path = dest_path / Path(url).name

    if f_path.exists():
        print('File already exists')
        if extract:
            extract_path = dest_path / Path(url).stem
            extract_files(f_path, extract_path, recursive=True,
                          remove_compressed=remove_compressed)
        return extract_path
    else:
        dest_path.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=10)
    total_sz = int(response.headers.get('content-length', 0))
    chunk_size = 1024

    pbar = tqdm(total=total_sz, unit='iB', unit_scale=True,
                desc=f'Downloading {url.split("/")[-1]}', dynamic_ncols=True)

    with open(f_path, 'wb') as file:
        for data in response.iter_content(chunk_size):
            pbar.update(len(data))
            file.write(data)

    pbar.close()

    if total_sz != 0 and pbar.n != total_sz:
        print('ERROR: Download failed')
        return None

    if extract:
        extract_path = dest_path / Path(url).stem
        extract_files(f_path, extract_path, recursive=True,
                      remove_compressed=remove_compressed)
        f_path = extract_path

    return f_path


def extract_files(f_path: str, dest_path: str, recursive: bool = False, remove_compressed: bool = False) -> None:
    """
    Extract files from a compressed file
    Args:
        f_path (str): Path to the compressed file
        dest_path (str): Path to extract the files
        recursive (bool): Extract files within the extracted folder
        remove_compressed (bool): Remove the compressed file after extraction
    Returns:
        None
    """

    f_path = Path(f_path)
    dest_path = Path(dest_path)

    if not f_path.exists():
        print('ERROR: File does not exist')
        return

    if not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)

    if f_path.suffix == '.tar':
        f = tarfile.open(f_path, 'r')
        members = f.getmembers()
    elif f_path.suffix == '.tar.gz' or f_path.suffix == '.tgz':
        f = tarfile.open(f_path, 'r:gz')
        members = f.getmembers()
    elif f_path.suffix == '.zip':
        f = zipfile.ZipFile(f_path, 'r')
        members = f.namelist()
    else:
        print('ERROR: Unsupported file format')
        return

    with tqdm(members, desc=f'Extracting {f_path.name}', dynamic_ncols=True) as pbar:
        for member in pbar:
            f.extract(member, dest_path)
            pbar.update(1)

    f.close()

    # Remove the original compressed file
    if f_path.exists() and remove_compressed:
        f_path.unlink()

    # Check if there are any other files to be extracted within the extracted folder
    if recursive:
        for file in dest_path.rglob('*.*'):
            if file == f_path:
                continue
            if file.is_file():
                if file.suffix == '.tar':
                    extract_files(file, dest_path, recursive=True,
                                  remove_compressed=remove_compressed)
                elif file.suffix == '.tar.gz' or file.suffix == '.tgz':
                    extract_files(file, dest_path, recursive=True,
                                remove_compressed=remove_compressed)
                elif file.suffix == '.zip':
                    extract_files(file, dest_path, recursive=True,
                                  remove_compressed=remove_compressed)
