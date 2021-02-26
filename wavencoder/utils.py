import os
from tqdm import tqdm
import urllib.request
import torchaudio

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False 


def _reporthook(t):
    """ ``reporthook`` to use with ``urllib.request`` that prints the process of the download.

    Uses ``tqdm`` for progress bar.

    **Reference:**
    https://github.com/tqdm/tqdm

    Args:
        t (tqdm.tqdm) Progress bar.

    Example:
        >>> with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # doctest: +SKIP
        ...   urllib.request.urlretrieve(file_url, filename=full_path, reporthook=reporthook(t))
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        Args:
            b (int, optional): Number of blocks just transferred [default: 1].
            bsize (int, optional): Size of each block (in tqdm units) [default: 1].
            tsize (int, optional): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner

def example_wav_file():
    filename = 'LDC93S1.wav'
    wav_url = 'https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav'
    if not os.path.exists(filename):
                    print(f'Downloading sample wav file from TIMIT Corpus - ({wav_url})', flush=True)
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                        urllib.request.urlretrieve(wav_url, filename, reporthook=_reporthook(t))
    
    waveform, sample_rate = torchaudio.load(filename)
    return waveform, sample_rate


def download_noise_dataset(path='./', sample_rate='16k', download_all=True, noise_envs=None):
    data_dict = {
        "kitchen" : f"https://zenodo.org/record/1227121/files/DKITCHEN_{sample_rate}.zip",
        "living" : f"https://zenodo.org/record/1227121/files/DLIVING_{sample_rate}.zip",
        "washing" : f"https://zenodo.org/record/1227121/files/DWASHING_{sample_rate}.zip",
        "field" : f"https://zenodo.org/record/1227121/files/NFIELD_{sample_rate}.zip",
        "park" : f"https://zenodo.org/record/1227121/files/NPARK_{sample_rate}.zip",
        "river" : f"https://zenodo.org/record/1227121/files/NRIVER_{sample_rate}.zip",
        "hallway" : f"https://zenodo.org/record/1227121/files/OMEETING_{sample_rate}.zip", 
        "meeting" : f"https://zenodo.org/record/1227121/files/OMEETING_{sample_rate}.zip",
        "office" : f"https://zenodo.org/record/1227121/files/OOFFICE_{sample_rate}.zip",
        "cafeter" : f"https://zenodo.org/record/1227121/files/PCAFETER_{sample_rate}.zip",
        "resto" : f"https://zenodo.org/record/1227121/files/PRESTO_{sample_rate}.zip",
        "station" : f"https://zenodo.org/record/1227121/files/PSTATION_{sample_rate}.zip",
        "metro" : f"https://zenodo.org/record/1227121/files/TMETRO_{sample_rate}.zip",
        "car" : f"https://zenodo.org/record/1227121/files/TCAR_{sample_rate}.zip",
        "bus" : f"https://zenodo.org/record/1227121/files/TBUS_{sample_rate}.zip",
        "traffic" : f"https://zenodo.org/record/1227121/files/STRAFFIC_{sample_rate}.zip"
    }

    if download_all:
        download_list = list(data_dict.keys())
    else:
        download_list = noise_envs
    
    for i_download in download_list:
        filename = data_dict[i_download].split('/')[-1]
        filepath = os.path.join(path, filename)
        file_url = data_dict[i_download]
        if not os.path.exists(filepath):
                    print(f'Downloading sample wav file from TIMIT Corpus - ({file_url})', flush=True)
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=filepath) as t:
                        urllib.request.urlretrieve(file_url, filepath, reporthook=_reporthook(t))



    