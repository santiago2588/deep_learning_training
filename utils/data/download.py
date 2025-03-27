from pathlib import Path
from tqdm import tqdm
import requests
import tarfile
import zipfile

DATA_URLS = {'air quality': 'https://archive.ics.uci.edu/static/public/360/air+quality.zip',
             'steel plates faults': 'https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip',
             'ARKOMA': 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/brg4dz8nbb-1.zip',
             'alphonso mangoes': 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/8sjny373pz-1.zip',
             'guava fruits': 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/fspx44mwfp-1.zip',
             'mango leaves': 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/nnh69sng8p-5.zip',
             'spot welding': 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/rwh8kjzdch-2.zip'
             }


def download_dataset(url: str, dest_path: str, extract: bool = False, remove_compressed: bool = False) -> Path:
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
                elif file.suffix == '.zip':
                    extract_files(file, dest_path, recursive=True,
                                  remove_compressed=remove_compressed)
