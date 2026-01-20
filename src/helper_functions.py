import requests
from pathlib import Path
from typing import Tuple, Dict, List, Union, Optional, Any
import logging
import os
from datetime import datetime
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that uses tqdm.write() to print above progress bars."""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(name: str,
                 log_to_file: bool = False,
                 log_dir: str = "logs",
                 level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that prints above tqdm progress bars and optionally logs to file.

    Parameters:
    - name (str): Name of the logger (typically __name__)
    - log_to_file (bool): Whether to also log to a file
    - log_dir (str): Directory to store log files (if log_to_file is True)
    - level (int): Logging level

    Returns:
    - logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs from root logger

    if not logger.handlers:
        # Tqdm-aware console handler
        console_handler = TqdmLoggingHandler()
        console_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
                                              datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


def read_disprot_fasta(file_loc: Union[str, Path]) -> Tuple[Dict[str, str], Dict[str, List[Union[int, str]]]]:
    """
    Reads CAID formatted FASTA files where entries are expected in blocks of 3:
    1. Header
    2. Sequence
    3. Annotation
    
    :param file_loc: Location of file
    :return: Tuple of dictionaries (sequences, annotations) keyed by header
    """
    file_path = Path(file_loc)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    disprot_sequences = {}
    disprot_annotation = {}
    
    current_header = None
    # We use a state tracker: 0=Expect Header, 1=Expect Seq, 2=Expect Annot
    state = 0 

    try:
        with file_path.open('r') as fn:
            for line_num, line in enumerate(fn, 1):
                line = line.strip()
                if not line: continue  # Skip empty lines

                if line.startswith(">"):
                    current_header = line
                    state = 1
                elif state == 1:
                    disprot_sequences[current_header] = line
                    state = 2
                elif state == 2:
                    # Parse annotation safely
                    try:
                        annot_data = ['-' if x == '-' else int(x) for x in line]
                    except ValueError:
                        # Fallback if char is neither '-' nor int compatible
                        print(f"Warning: Unexpected character in annotation at line {line_num}")
                        annot_data = [x for x in line]
                    
                    disprot_annotation[current_header] = annot_data
                    state = 0  # Reset to expect next header
                else:
                    print(f"Warning: formatting issue at line {line_num}. Expected header.")

    except Exception as e:
        raise ValueError(f"Error parsing DisProt file: {e}")

    assert len(disprot_sequences) == len(disprot_annotation), 'Length of sequences does not match length of annotations'

    return disprot_sequences, disprot_annotation


def read_fasta(file_location: Union[str, Path], 
               data_type: type = str, 
               split: Optional[str] = None) -> Dict[str, Any]:
    """
    Generic FASTA reader that handles multi-line sequences and specific data types.
    
    :param file_location: Path to the FASTA file
    :param data_type: The type to convert sequence data into (default: str)
    :param split: Delimiter if data needs splitting (e.g., space separated numbers)
    :return: Dictionary of headers and parsed sequences
    """
    file_path = Path(file_location)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    fasta = {}
    header = None
    # Use a list to accumulate sequence parts (faster than string concatenation)
    sequence_buffer = []

    def save_buffer(hdr, buf):
        if hdr and buf:
            if data_type == str:
                fasta[hdr] = "".join(buf)
            else:
                # If non-string, flatten the buffer list
                flat_list = [item for sublist in buf for item in sublist]
                fasta[hdr] = flat_list

    with file_path.open('r') as fn:
        for line in fn:
            line = line.strip()
            if not line: continue

            if line.startswith('>'):
                # Save previous entry before starting new one
                save_buffer(header, sequence_buffer)
                
                header = line.lstrip('>')
                sequence_buffer = []
            else:
                if header is None:
                    continue # Skip content before first header
                
                if data_type == str:
                    sequence_buffer.append(line)
                else:
                    # Handle non-string data types (e.g., numeric scores)
                    parts = line.split(split)
                    try:
                        converted = [data_type(x) for x in parts if x]
                        sequence_buffer.append(converted)
                    except ValueError:
                         print(f"Warning: Could not convert data to {data_type} for {header}")

        # Save the last entry
        save_buffer(header, sequence_buffer)

    return fasta


def uniprot_download(accession: str, out_dir: str = 'data/seq', overwrite: bool = False) -> str:
    """
    Downloads a FASTA file from UniProt by accession ID.
    
    :param accession: UniProt Accession ID (e.g., 'Q32P44')
    :param out_dir: Directory to save the file
    :param overwrite: If True, redownloads even if file exists
    :return: Path to the downloaded file
    """
    path = Path(out_dir)
    # Create directory if it doesn't exist (parents=True handles nested dirs like 'data/seq')
    path.mkdir(parents=True, exist_ok=True)
    
    file_path = path / f"{accession}.fasta"

    if file_path.exists() and not overwrite:
        return str(file_path)

    url = f'https://rest.uniprot.org/uniprotkb/{accession}.fasta'
    
    try:
        resp = requests.get(url, timeout=30)
        # Raise error for 4xx or 5xx status codes
        resp.raise_for_status()
        
        with file_path.open('w+') as fw:
            fw.write(resp.text)
            
    except requests.exceptions.HTTPError as err:
        raise SystemError(f"Failed to download {accession}: {err}")
    except requests.exceptions.RequestException as e:
        raise SystemError(f"Network error while downloading {accession}: {e}")

    return str(file_path)


if __name__ == '__main__':
    # Example usage
    try:
        saved_path = uniprot_download('Q32P44')
        print(f"File saved to: {saved_path}")
        
        # Testing the fasta reader on the downloaded file
        sequences = read_fasta(saved_path)
        print(f"Read {len(sequences)} sequences.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

    caid2_seq, caid2_annot = read_disprot_fasta('data/caid2/binding.fasta')
    print(len(caid2_seq))

