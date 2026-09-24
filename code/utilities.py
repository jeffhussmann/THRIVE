import shutil
from pathlib import Path

def flatten_basespace_directory_structure(data_dir):
    ''' Basespace downloads projects into a nested directory structure.
    This moves all fastqs into the top level dir, then cleans up the subdirs.
    '''

    data_dir = Path(data_dir)
    # Flatten and clean up basespace download directory structure.

    for fn in sorted(data_dir.rglob('*.fastq.gz')):
        shutil.move(fn, data_dir)

    for sub_dir in data_dir.iterdir():
        if sub_dir.is_dir():
            sub_dir.rmdir()