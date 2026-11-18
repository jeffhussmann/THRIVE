import shutil
from pathlib import Path

import pandas as pd

def flatten_basespace_directory_structure(data_dir):
    ''' Basespace downloads projects into a nested directory structure.
    This moves all fastqs into the top level dir, then cleans up the subdirs.
    '''

    data_dir = Path(data_dir)

    for fn in sorted(data_dir.rglob('*.fastq.gz')):
        shutil.move(fn, data_dir)

    for sub_dir in data_dir.iterdir():
        if sub_dir.is_dir():
            sub_dir.rmdir()

def load_registries(base_dir):
    base_dir = Path(base_dir)

    guides = pd.read_csv(base_dir / 'registries' / 'guides.csv', index_col='Name').fillna('')
    guides.columns = guides.columns.str.strip()

    variants = pd.read_csv(base_dir / 'registries' / 'variants.csv', index_col='Name').fillna('')
    variants.index = variants.index.str.strip()

    editors = pd.read_csv(base_dir / 'registries' / 'editors.csv', index_col='F#').fillna('')
    editors.index = editors.index.str.strip()

    return guides, variants, editors

def process_sample_sheet(base_dir, incoming_fn, variant_to_which_read=None):
    PAM_to_BE_effector = {
        'NGG': 'SpCas9_D10A',
        'NGA': 'SpCas9-NGA',
        'NRRH': 'SpCas9-NRRH_D10A',
        'NRTH': 'SpCas9-NRTH_D10A',
        'NRCH': 'SpCas9-NRCH_D10A',
        'SpRY': 'SpRY_D10A',
        'eSpGC': 'eSpGC_D10A',
        'eSpGA': 'eSpGA_D10A',
        'SpGA': 'eSpGA_D10A',
        'eSpGT': 'eSpGT_D10A',
        'LWRYEK': 'SpCas9(LWRYEK)_D10A',
        'LWKYQS': 'SpCas9(LWKYQS)_D10A',
        'MKRCMV': 'SpCas9(MKRCMV)_D10A',
        'MKRCKS': 'SpCas9(MKRCKS)_D10A',
        'MRRWMR': 'SpCas9(MRRWMR)_D10A',
        'ARGIMR': 'SpCas9(ARGIMR)_D10A',
        'NG': 'SpCas9-NG_D10A',
        'eVRQR': 'SpCas9_eVRQR_D10A',
        'eMQKSER': 'SpCas9(eMQKSER)_D10A',
        'KWRQLC': 'SpCas9(KWRQLC)_D10A',
        'MRKCRS': 'SpCas9(MRKCRS)_D10A',
        'VRAVQL': 'SpCas9(VRAVQL)_D10A',
        'NWA': 'SpCas9-NWA_D10A',
        '': 'SpCas9',
    }

    PAM_to_PE_effector = {
        'NGG': 'SpCas9_H840A',
        'NGA': 'SpCas9-NGA_H840A',
        '': 'SpCas9_H840A',
    }

    guides, variants, editors = load_registries(base_dir)

    sgRNA_to_variant = guides['Variant Name'].map(str.strip)

    if not guides.index.is_unique:
        raise ValueError

    incoming_df = pd.read_csv(incoming_fn).fillna('')

    sample_sheet = {}

    for _, row in incoming_df.iterrows():
        name = row['Sample ID']

        R1_fn = f'{row['Fastq ID']}_L001_R1_001.fastq.gz'
        R2_fn = f'{row['Fastq ID']}_L001_R2_001.fastq.gz'

        primers = ';'.join([row['Primer 1'], row['Primer 2']])

        sgRNAs = row['sgRNA']

        if row['F#'] != '':
            editor_row = editors.loc[row['F#']]
            editor = editor_row['Name']
            PAM_preference = editor_row['PAMPreference']

            deaminase = editor.rsplit('-', 1)[0]

            if editor_row['Type'] == 'PE':
                effector = PAM_to_PE_effector[PAM_preference]
            else:
                effector = PAM_to_BE_effector[PAM_preference]

        else:
            editor = ''
            deaminase = ''

            effector = PAM_to_BE_effector['']

        sgRNAs_list = sgRNAs.split(';')

        variant_names = set(variant for sgRNA in sgRNAs_list if (variant := sgRNA_to_variant.get(sgRNA, '')) != '')

        if len(variant_names) == 0:
            variant = ''

        elif len(variant_names) == 1:
            variant_name = list(variant_names)[0]
            variant = variants.loc[variant_name, 'HGVS (genomic)']

        else:
            raise ValueError

        sample_sheet[name] = {
            'platform': 'illumina',
            'experiment_type': 'prime_editing',
            'genotype': row['Genotype'],
            'primers': primers,
            'sgRNAs': sgRNAs,
            'effector': effector,
            'condition:deaminase': deaminase,
            'condition:variant': variant,
            'condition:editor': editor,
        }

        if variant_to_which_read is None:
            sample_sheet[name].update({
                'R1': R1_fn,
                'R2': R2_fn,
            })

        else:
            which_read = variant_to_which_read.get(variant_name, 'R1')

            if which_read == 'R1':
                fn = R1_fn
            else:
                fn = R2_fn

            sample_sheet[name].update({
                'R1': fn,
            })

        sample_sheet[name].update(row[row.index.str.startswith('condition:')])

    sample_sheet = pd.DataFrame(sample_sheet).T
    sample_sheet.index.name = 'sample_name'

    return sample_sheet, variants
