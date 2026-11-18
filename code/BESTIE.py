import pandas as pd
import hits.utilities
import pooled_pegRNA_screens

import base_editing_screens

from hits import sw, fastq, adapters
from collections import Counter
import itertools

memoized_property = hits.utilities.memoized_property

class PooledScreenVariant(base_editing_screens.Screen):
    def __init__(self,
                 base_dir,
                 batch_names,
                 var_g,
                 gene,
                 ignore_indels=False
                ):

        self.base_dir = base_dir
        self.batch_names = batch_names
        self.var_g = var_g
        self.gene = gene

        self.heterozygous = False
        self.install = False

        self.specific_unedited_variant = None

        self.variant_primers = 'n/a'

        self.ignore_indels = ignore_indels

    @memoized_property
    def pools(self):
        return pooled_pegRNA_screens.get_all_pools(self.base_dir, batch_names=self.batch_names)

    @memoized_property
    def first_pool(self):
        batch_name = sorted(self.pools)[0]
        sample_name = sorted(self.pools[batch_name])
        return self.pools[batch_name][sample_name]

    @memoized_property
    def variant_name(self):
        library = self.first_pool.library

        var_g_to_variant_name = dict(zip(library.elements_df['var_g'], library.elements_df['variant_name']))

        return var_g_to_variant_name[self.var_g]

    @memoized_property
    def exps(self):
        exps = {}

        for batch_name, pools in self.pools.items():
            for sample_name, pool in pools.items():
                element_names = pool.library.elements_df.query('var_g == @self.var_g').index
                for element_name in element_names:
                    exps[batch_name, sample_name, element_name] = pool.experiment_from_identifier_fields(element_name=element_name)

        return exps

    exp_keys = ['batch_name', 'sample_name', 'element_name']

    def row_to_outcome_counts(self, row):
        pool = self.pools[row['batch_name']][row['sample_name']]
        partition_name = pool.element_name_to_partition_name(row['element_name'])
        outcome_counts = pool.outcome_counts_store(partition_name)[row['element_name'], 'consistent']
        return outcome_counts

    @memoized_property
    def sample_sheet(self):
        sample_sheet = {}

        for (batch_name, sample_name, element_name), exp in self.exps.items():
            sample_sheet[batch_name, sample_name, element_name] = {
                'condition:variant': self.var_g if 'NTC' not in exp.element.row['original_name'] else '',
                'condition:deaminase': exp.experiment_group.details['editor'],
                'condition:editor': exp.experiment_group.details['editor'],
                'gene': exp.element.row['gene'],
                'primers': self.variant_primers,
                'sgRNAs': exp.element.row['original_name'],
                'PAM': 'NNN',
                'editor': exp.experiment_group.details['editor'],
                'variant_name': exp.element.row['variant_name'],
            }

        sample_sheet = pd.DataFrame(sample_sheet).T
        sample_sheet.index.names = ['batch_name', 'sample_name', 'element_name']

        return sample_sheet

def extract_lengths(base_dir):
    all_lengths = {}

    pools = pooled_pegRNA_screens.get_all_pools(base_dir)

    for batch_name, batch_pools in pools.items():
        for pn, pool in batch_pools.items():
            print(pn)

            read_pairs = fastq.read_pairs(pool.fns['R1'], pool.fns['R2'], standardize_names=True, up_to_space=True)
            
            before_R1 = adapters.primers['truseq']['R1']
            before_R2 = adapters.primers['truseq']['R2']
            
            stitched_lengths = Counter()
            
            for R1, R2 in itertools.islice(read_pairs, 20000):
                if R1.name != R2.name:
                    raise ValueError
            
                stitched = sw.stitch_read_pair(R1, R2, before_R1, before_R2, indel_penalty=-1000)
            
                stitched_lengths[len(stitched)] += 1
            
            lengths_array = hits.utilities.counts_to_array(stitched_lengths)
            
            all_lengths[batch_name, pool.details['editor']] = lengths_array

    all_lengths = pd.DataFrame(all_lengths).fillna(0).astype(int)

    return all_lengths