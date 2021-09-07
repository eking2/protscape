import numpy as np
import pandas as pd
import requests
from typing import Union, Optional
from pathlib import Path
import tarfile

# gap (-), unknown (X)
AA_LIST = "ACDEFGHIKLMNPQRSTVWXY-"
URLS = {
    "stability": "http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/stability.tar.gz",
    "fluorescence": "http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/fluorescence.tar.gz",
    "envision": "https://dl.fbaipublicfiles.com/fair-esm/examples/P62593.fasta",
}


def seq_to_ohe(seq: str, pad: Optional[int] = None) -> np.ndarray:

    """Convert amino acid sequence to one-hot format.

    Parameters
    ----------
    seq : str
        Amino acid sequences
    pad : Optional[int]
        Length to right pad up to, use gap char (-)

    Returns
    -------
    ohe : np.ndarray
        Sequence in one-hot format, shape: (seq length, 22)
    """

    # add padding
    if pad is not None:
        to_add = pad - len(seq)
        seq = seq + ('-' * to_add)

    # to label
    labels = [AA_LIST.index(c) for c in seq]

    # ohe
    ohe = np.eye(len(AA_LIST))[labels]

    return ohe


def ohe_to_seq(ohe: np.ndarray) -> str:

    """Convert one-hot array to amino acid sequence.

    Parameters
    ----------
    ohe : np.ndarray
        Sequence in one-hot format, shape: (seq length, 22)

    seq : str
        Amino acid sequence
    """

    # indices
    indices = np.argmax(ohe, axis=1)

    # to str
    seq = "".join([AA_LIST[i] for i in indices])

    return seq


def download_tape(dataset: str, data_dir: Union[str, Path] = "./data") -> None:

    """Download TAPE dataset (GFP or stability) in json format.

    Parameters
    ----------
    dataset : str
        Name of dataset
    data_dir : Union[str, Path]
        Path to save dataset
    """

    assert dataset in [
        "stability",
        "fluorescence",
    ], f'invalid dataset: {dataset}, choices are ["stability", "fluorescence"]'

    # check if already downloaded
    url = URLS[dataset]
    fn = url.split("/")[-1]
    dl = Path(data_dir, fn)
    if dl.exists():
        print(f"{dataset} already downloaded")
        return

    # download tar.gz
    print(f"Downloading {dataset}")
    r = requests.get(url)
    r.raise_for_status()
    Path(dl).write_bytes(r.content)

    # untar
    with tarfile.open(dl, "r") as tar:
        tar.extractall(Path(data_dir))


def download_envision(data_dir: Union[str, Path]) -> None:

    """Download Envision dataset FASTA.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to save dataset
    """

    # check if already downloaded
    url = URLS["envision"]
    fn = url.split("/")[-1]
    dl = Path(data_dir, fn)
    if dl.exists():
        print(f"envision already downloaded")
        return

    # download fasta
    print(f"Downloading envision")
    r = requests.get(url)
    r.raise_for_status()
    Path(dl).write_text(r.text)
