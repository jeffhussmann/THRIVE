import logging

import bokeh.palettes
import Bio.SeqUtils
import hgvs.easy
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyensembl
import seaborn as sns
import varcode

import hits.genomes
import hits.utilities

memoized_property = hits.utilities.memoized_property

logger = logging.getLogger(__name__)

ensembl = pyensembl.EnsemblRelease(species='human')

genomic_sequence_fetcher = hits.genomes.build_region_fetcher('/home/jah/indices/hg38/fasta')

MANE_transcripts = pd.read_csv('/home/jah/projects/THRIVE/data/MANE.GRCh38.v1.5.summary.txt', sep='\t')

def get_MANE_row(gene_name):
    rows = MANE_transcripts.query('symbol == @gene_name')

    if len(rows) != 1:
        logger.warning(f'{gene_name} has {len(rows)} rows')
        rows = rows.query('MANE_status == "MANE Plus Clinical"')

    if len(rows) != 1:
        raise ValueError

    MANE_row = rows.iloc[0]

    return MANE_row

def simple_name_to_hgvs(simple_name):
    gene_name, change, *rest = simple_name.split()
    change = change.strip(',')

    MANE_row = get_MANE_row(gene_name)

    RefSeq_nuc = MANE_row['RefSeq_nuc']
    Ensembl_nuc = MANE_row['Ensembl_nuc'].split('.')[0]

    var_c = hgvs.easy.parse(f'{RefSeq_nuc}:{change}')
    
    var_g = hgvs.easy.c_to_g(var_c)
    
    var_g.validate()

    return var_g, var_c

def maximize_text_visibility(color):
    # Taken directly from seaborn's approach to annot in heatmaps.
    lum = sns.utils.relative_luminance(color)
    text_color = '.15' if lum > .408 else 'w'

    return text_color

class AdenineDeaminase:
    def __init__(self, name, optimal_window=None, bystander_window=None):
        self.name = name

        if optimal_window is None:
            optimal_window = [5, 5]

        if bystander_window is None:
            bystander_window = [1, 20]

        self.optimal_window = optimal_window
        self.bystander_window = bystander_window

        optimal_start, optimal_end = self.optimal_window
        optimal_positions = np.arange(optimal_start, optimal_end + 1)

        bystander_start, bystander_end = self.bystander_window
        bystander_positions = np.arange(bystander_start, bystander_end + 1)

        self.offsets_from_PAM = {
            'optimal_window': [-20 + p - 1 for p in optimal_positions],
            'bystander_window': [-20 + p - 1 for p in bystander_positions],
            'entire_protospacer': [-20 + p - 1 for p in np.arange(1, 20 + 1)], 
        }

deaminase_specs = {
    'ABE8e': {
        'optimal_window': [4, 8],
        'bystander_window': [3, 11],
    },

    'ABE8.20': {
        'optimal_window': [4, 7],
        'bystander_window': [3, 9],
    },

    'ABE7.10-HRHSK': {
        'optimal_window': [5, 7],
        'bystander_window': [4, 8],
    },
}

class Editor:
    def __init__(self, deaminase_name, PAM):
        specs = deaminase_specs.get(deaminase_name, {})
        self.deaminase = AdenineDeaminase(deaminase_name, **specs)
        self.PAM = PAM
        
        self.name = f'{self.deaminase.name}-{self.PAM}'

class BaseEditableVariant:
    def __init__(self, var_g, gene_name=None, install=False):
        if isinstance(var_g, str):
            var_g = hgvs.easy.parse(var_g)

        self.var_g = var_g

        self.gene_name = gene_name

        self.install = install

        self.contig = str(int(self.var_g.ac.split('.')[0].split('_')[-1]))
        if self.contig == '23':
            self.contig = 'X'

        self.variant_genomic_1_p = self.var_g.posedit.pos.start.base

        ref = self.var_g.posedit.edit.ref
        alt = self.var_g.posedit.edit.alt
        
        self.variant = varcode.Variant(self.contig, self.variant_genomic_1_p, ref, alt)

        edit = str(self.var_g.posedit.edit)

        if not self.install:
            if edit == 'G>A':
                self.strand = '+'
                self.type = 'ABE'

            elif edit == 'C>T':
                self.strand = '-'
                self.type = 'ABE'

            elif edit == 'T>C':
                self.strand = '+'
                self.type = 'CBE'

            elif edit == 'A>G':
                self.strand = '-'
                self.type = 'CBE'

            else:
                raise ValueError(edit)

        else:
            if edit == 'G>A':
                self.strand = '-'
                self.type = 'CBE'

            elif edit == 'C>T':
                self.strand = '+'
                self.type = 'CBE'

            elif edit == 'T>C':
                self.strand = '-'
                self.type = 'ABE'

            elif edit == 'A>G':
                self.strand = '+'
                self.type = 'ABE'

            else:
                raise ValueError(edit)

        if self.strand == '+':
            window_before = 15
            window_after = 25
        else:
            window_before = 25
            window_after = 15

        self.other_strand = {'+': '-', '-': '+'}[self.strand]

        # TODO: handle if close to edge
    
        window_start = self.variant_genomic_1_p - 1 - window_before
        window_end = self.variant_genomic_1_p - 1 + window_after + 1
        window_sequence = genomic_sequence_fetcher(f'chr{self.contig}', window_start, window_end).upper()

        # If a correction, switch to the alt. If an install, don't.

        if self.install:
            self.target_strand_window_sequence = window_sequence
        else:
            self.target_strand_window_sequence = window_sequence[:window_before] + self.var_g.posedit.edit.alt + window_sequence[window_before + 1:]

        if self.strand == '+':
            self.variant_p_in_window = window_before

        else:
            self.target_strand_window_sequence = hits.utilities.reverse_complement(self.target_strand_window_sequence)
            self.variant_p_in_window = window_after

    def get_effect(self, variant):
        effects = variant.effects()
        
        effect = effects.top_priority_effect_per_transcript_id()[self.ensembl_nuc]

        return effect

    @memoized_property
    def effect(self):
        return self.get_effect(self.variant)

    @memoized_property
    def MANE_row(self):
        return get_MANE_row(self.gene_name)

    @memoized_property
    def refseq_nuc(self):
        return self.MANE_row['RefSeq_nuc']

    @memoized_property
    def ensembl_nuc(self):
        return self.MANE_row['Ensembl_nuc'].split('.')[0]

    def window_p_to_genomic_1_p(self, window_p):
        if self.strand == '+':
            return self.variant_genomic_1_p - self.variant_p_in_window + window_p
        else: 
            return self.variant_genomic_1_p + self.variant_p_in_window - window_p

    def genomic_1_p_to_window_p(self, genomic_1_p):
        if self.strand == '+':
            return self.variant_p_in_window + (genomic_1_p - self.variant_genomic_1_p)
        else: 
            return self.variant_p_in_window - (genomic_1_p - self.variant_genomic_1_p)

    @property
    def relevant_transcripts(self):
        return hgvs.easy.am38.relevant_transcripts(self.var_g)

    @property
    def var_c(self):
        return hgvs.easy.g_to_c(self.var_g, self.refseq_nuc)

    @property
    def MANE_transcript(self):
        return ensembl.transcript_by_id(self.ensembl_nuc)

    @property
    def gene(self):
        return ensembl.gene_name_of_gene_id(self.MANE_transcript.gene_id)

    @property
    def label(self):
        return f'{self.var_g}\n{self.var_c}\n{self.gene} {self.effect.short_description}'

    @property
    def simple_label(self):
        return f'{self.gene} {self.var_c.posedit} ({self.effect.short_description})'

    @hits.utilities.memoized_property
    def possible_bystander_effects(self):
        possible_bystander_effects = {}
        
        for window_p, b in enumerate(self.target_strand_window_sequence):
            if window_p == self.variant_p_in_window:
                continue
                
            genomic_1_p = self.window_p_to_genomic_1_p(window_p)
    
            # If bottom strand, rc the variant
    
            if self.strand == '+':
                ref = 'A'
                alt = 'G'
            else: # self.strand == '-'
                ref = 'T'
                alt = 'C'
        
            if b == 'A':
                variant = varcode.Variant(self.contig, genomic_1_p, ref, alt)

                possible_bystander_effects[window_p] = self.get_effect(variant)
                
        return possible_bystander_effects

    def PAMs(self, editor):
        pattern, *PAM_starts = Bio.SeqUtils.nt_search(self.target_strand_window_sequence, editor.PAM)

        PAMs = {PAM_start: self.PAM(PAM_start, editor) for PAM_start in PAM_starts}

        return PAMs

    def PAM(self, PAM_start_window_p, editor):
        bystander_effects = []
    
        for possible_bystander_offset in editor.deaminase.offsets_from_PAM['entire_protospacer']:
            window_p = PAM_start_window_p + possible_bystander_offset
    
            if window_p in self.possible_bystander_effects:
                window_name = None
                for possible_window_name in ['optimal_window', 'bystander_window', 'entire_protospacer']:
                    if possible_bystander_offset in editor.deaminase.offsets_from_PAM[possible_window_name]:
                        window_name = possible_window_name
                        break

                if window_name is None:
                    raise ValueError

                effect = self.possible_bystander_effects[window_p]
                bystander_effects.append({
                    'window_name': window_name,
                    'effect': effect,
                    'bystander_offset_from_PAM': window_p - PAM_start_window_p,
                    'bystander_position_in_protospacer': window_p - (PAM_start_window_p - 21),
                    'actual': possible_bystander_offset,
                    'window_p': window_p,
                    'genomic_1_p': self.window_p_to_genomic_1_p(window_p),
                })

        variant_offset_from_PAM = self.variant_p_in_window - PAM_start_window_p
        window_name = 'outside_protospacer'
        for possible_window_name in ['optimal_window', 'bystander_window', 'entire_protospacer']:
            if variant_offset_from_PAM in editor.deaminase.offsets_from_PAM[possible_window_name]:
                window_name = possible_window_name
                break
    
        PAM = {
            'window_name': window_name,
            'PAM_start': PAM_start_window_p,
            'protospacer': self.target_strand_window_sequence[PAM_start_window_p - 20:PAM_start_window_p],
            'PAM': self.target_strand_window_sequence[PAM_start_window_p:PAM_start_window_p + len(editor.PAM)],
            'protospacer+PAM': self.target_strand_window_sequence[PAM_start_window_p - 20:PAM_start_window_p + len(editor.PAM)],
            'bystander_effects': bystander_effects,
            'strand': self.strand,
            'variant_offset_from_PAM': self.variant_p_in_window - PAM_start_window_p,
            'variant_position_in_protospacer': self.variant_p_in_window - (PAM_start_window_p - 21),
            'editor': editor,
        }

        return PAM

    def PAM_from_editing_strategy(self, editing_strategy, editor):
        if editing_strategy.primary_protospacer is None:
            PAM = None

        else:
            # First convert from target coords to genomic coords,
            # then genomic coords to windows coords.

            feature = editing_strategy.PAM_features[f'{editing_strategy.primary_protospacer}_PAM']

            if feature.strand == '+':
                PAM_start = feature.start
            else:
                PAM_start = feature.end
            
            PAM_start += editing_strategy.offset_for_genomic_1(self.var_g)
    
            PAM_start_window_p = self.genomic_1_p_to_window_p(PAM_start)

            PAM = self.PAM(PAM_start_window_p, editor)

        return PAM

    def PAMs_with_variant_in_optimal_window(self, editor):
        return {position: PAM for position, PAM in self.PAMs(editor).items() if PAM['window_name'] == 'optimal_window'}

    def PAMs_with_variant_in_bystander_window(self, editor):
        return {position: PAM for position, PAM in self.PAMs(editor).items() if PAM['window_name'] in {'optimal_window', 'bystander_window'}}

    def PAMs_with_variant_in_protospacer(self, editor):
        return {position: PAM for position, PAM in self.PAMs(editor).items() if PAM['window_name'] in {'optimal_window', 'bystander_window', 'entire_protospacer'}}

    def usable_PAMs(self, editor):
        PAMs = self.PAMs_with_variant_in_optimal_window(editor)
        usable_PAMs = {position: PAM for position, PAM in PAMs.items() if PAM_is_usable(PAM)}
        return usable_PAMs

    def has_usable_PAM(self, editor):
        return len(self.usable_PAMs(editor)) > 0

effect_type_to_color = {
    (varcode.effects.Silent, False): 'orange',
    (varcode.effects.Silent, True): bokeh.palettes.Colorblind8[1],

    (varcode.effects.Substitution, False): 'red',
    (varcode.effects.Substitution, True): bokeh.palettes.Colorblind8[5],

    (type(None), False): 'grey', # represents non-A
    (type(None), True): 'grey', # represents non-A

    ('other', False): 'purple',
    ('other', True): bokeh.palettes.Colorblind8[0],

    ('pathogenic', False): 'green',
    ('pathogenic', True): bokeh.palettes.Colorblind8[3],
}

effect_type_to_cmap = {}

for (effect_type, colorblind), color in effect_type_to_color.items():
    if colorblind:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(f'{effect_type}_cmap', ['white', color])
        effect_type_to_cmap[effect_type, True] = cmap
    else:
        cmap_name = f'{color[0].upper()}{color[1:]}s'
        cmap = plt.get_cmap(cmap_name)

    effect_type_to_cmap[effect_type, colorblind] = cmap

def effect_to_color(effect, colorblind=False):
    return effect_type_to_color.get((type(effect), colorblind), effect_type_to_color['other', colorblind])

def effect_to_cmap(effect, colorblind=False):
    return effect_type_to_cmap.get((type(effect), colorblind), effect_type_to_cmap['other', colorblind])

def PAM_is_usable(PAM):
    def is_unacceptable(effect):
        in_window = effect['window_name'] in ['bystander_window', 'optimal_window']
        bad_effect = not isinstance(effect['effect'], varcode.effects.Silent)
        return in_window and bad_effect
        
    return not any(is_unacceptable(effect) for effect in PAM['bystander_effects'])

class VariantDiagram:

    transcript_y = 0.8
    
    def __init__(self, variant, colorblind=False):
        self.variant = variant

        self.colorblind = colorblind

        min_window_p = 0
        max_window_p = len(self.variant.target_strand_window_sequence) - 1
        self.x_min, self.x_max = sorted(map(self.variant.window_p_to_genomic_1_p, [min_window_p, max_window_p]))

        width = (self.x_max - self.x_min) * 10 / 40
        self.fig, self.ax = plt.subplots(figsize=(width, 4))

    def draw_sequence(self):
        ax = self.ax
        
        for window_p, target_strand_b in enumerate(self.variant.target_strand_window_sequence):
            if window_p == self.variant.variant_p_in_window:
                color = 'tab:green'

                weight = 'bold'
                other_weight = 'normal'
                
            elif window_p in self.variant.possible_bystander_effects:
                effect = self.variant.possible_bystander_effects[window_p]

                color = effect_to_color(effect, colorblind=self.colorblind)

                weight = 'bold'
                other_weight = 'normal'
                
            else:
                color = 'black'
                
                weight = 'normal'
                other_weight = 'normal'

            colors = {
                self.variant.strand: color,
                self.variant.other_strand: 'black',
            }

            weights = {
                self.variant.strand: weight,
                self.variant.other_strand: other_weight,
            }

            genomic_p = self.variant.window_p_to_genomic_1_p(window_p)
        
            kwargs = dict(
                xy=(genomic_p, 0),
                xycoords='data',
                textcoords='offset points',
                fontname='DejaVu Sans',
                size=8,
                ha='center',
            )

            bs = {
                self.variant.strand: target_strand_b,
                self.variant.other_strand: hits.utilities.complement(target_strand_b),
            }
            
            ax.annotate(bs['+'],
                        xytext=(0, 2),
                        va='bottom',
                        color=colors['+'],
                        weight=weights['+'], 
                        **kwargs,
                       )
        
            ax.annotate(bs['-'],
                        xytext=(0, -2),
                        va='top',
                        color=colors['-'],
                        weight=weights['-'], 
                        **kwargs,
                       )
        
            if window_p == self.variant.variant_p_in_window:
                x0 = self.variant.variant_genomic_1_p - 0.45
                x1 = self.variant.variant_genomic_1_p + 0.45
                y0 = -0.55
                y1 = 0.55
            
                path = [
                    [x0, y0],
                    [x0, y1],
                    [x1, y1],
                    [x1, y0],
                ]
            
                patch = plt.Polygon(path,
                                    closed=True,
                                    color='tab:green',
                                    fill=False,
                                   )
                ax.add_patch(patch)
        
        ax.set_xlim(self.x_min - 0.5, self.x_max + 0.5)
        
        ax.set_ylim(-5, 5)

    def draw_rect(self, x0, x1, y0, y1,
                  alpha=1,
                  color='black',
                  fill=True,
                  clip_to_window=True,
                  line_width=1,
                  label=None,
                  label_offset=(0, 3),
                  label_corner='upper left',
                 ):
        
        if clip_to_window:
            window_left, window_right = self.x_min, self.x_max

            if x0 > window_right or x1 < window_left:
                return

            x0 = max(x0, window_left - 0.5)
            x1 = min(x1, window_right + 0.5)

        path = [
            [x0, y0],
            [x0, y1],
            [x1, y1],
            [x1, y0],
        ]

        patch = plt.Polygon(path,
                            fill=fill,
                            closed=True,
                            alpha=alpha,
                            color=color,
                            clip_on=False,
                            linewidth=0 if fill else line_width,
                           )

        self.ax.add_patch(patch)

        if label is not None:
            if 'upper' in label_corner:
                y = y1
                va = 'bottom'
            elif 'lower' in label_corner:
                y = y0
                va = 'top'
            else:
                raise ValueError(label_corner)

            if 'right' in label_corner:
                x = x1
                ha = 'right'
            elif 'left' in label_corner:
                x = x0
                ha = 'left'
            else:
                raise ValueError(label_corner)

            self.ax.annotate(label,
                             xy=(x, y),
                             xytext=label_offset,
                             textcoords='offset points',
                             annotation_clip=False,
                             size=6, 
                             weight='bold',
                             ha=ha,
                             va=va,
                            )

    def draw_transcript(self, transcript):
        # exon_intervals are 1-based and inclusive
        exon_intervals_genomic = [
            (start_1, end_1)
            for start_1, end_1 in transcript.exon_intervals
        ]
    
        if transcript.strand == '-':
            exon_intervals_genomic = exon_intervals_genomic[::-1]
    
        intron_intervals_genomic = [
            (first[1] + 1, second[0] - 1)
            for first, second in hits.utilities.pairwise(exon_intervals_genomic)
        ]
    
        y = self.transcript_y
        height = 0.25

        gene = ensembl.gene_name_of_gene_id(transcript.gene_id)
    
        for exon_i, (start, end) in enumerate(exon_intervals_genomic):
            if transcript.strand == '-':
                exon_number = len(exon_intervals_genomic) - exon_i
            else:
                exon_number = exon_i + 1

            # Place label on corner farthest from variant.
            if abs(start - self.variant.variant_genomic_1_p) < abs(end - self.variant.variant_genomic_1_p):
                label_corner = 'upper right'
            else:
                label_corner = 'upper left'

            self.draw_rect(start - 0.5, end + 0.5, y - 0.5 * height, y + 0.5 * height,
                           0.5,
                           label=f'{gene} exon {exon_number}',
                           label_offset=(0, 12),
                           label_corner=label_corner,
                          )
    
        for start, end in intron_intervals_genomic:
            self.draw_rect(start - 0.5, end + 0.5, y - 0.25 * height, y + 0.25 * height,
                           0.5,
                           color='mediumpurple',
                          )
    
        amino_acids = extract_amino_acids(transcript)
    
        for amino_acid in amino_acids:
            self.draw_amino_acid(amino_acid)
    
    def draw_amino_acid(self, amino_acid, clip_to_window=True):
        y = self.transcript_y
        height = 0.3
    
        y0 = y - 0.5 * height
        y1 = y + 0.5 * height
    
        if amino_acid['strand'] == '+':
            start_offset = -0.5
            end_offset = 0.5
        else:
            start_offset = 0.5
            end_offset = -0.5

        colors = {
            'facecolor': 'lightgrey',
            'edgecolor': 'black',
            'text': 'black',
        }

        letter_to_draw = amino_acid['letter']
        weight = 'normal'
        
        for connected_block in amino_acid['connected_blocks']:
            position = amino_acid['position_1']

            if self.variant.variant_genomic_1_p in connected_block:
                if isinstance(self.variant.effect, (varcode.effects.Substitution, varcode.effects.PrematureStop)):
                    if self.variant.effect.aa_mutation_start_offset + 1 != position:
                        raise ValueError
                    if self.variant.effect.aa_ref != amino_acid['letter']:
                        raise ValueError

                    letter_to_draw = self.variant.effect.aa_alt

                colors = {
                    'facecolor': 'tab:green',
                    'edgecolor': 'black',
                    'text': 'black',
                }

                if letter_to_draw == '':
                    letter_to_draw = '*'

                weight = 'bold'

                self.ax.annotate(str(position),
                                xy=(np.mean(connected_block), y1),
                                xycoords='data',
                                xytext=(0, 1),
                                textcoords='offset points',
                                ha='center',
                                va='bottom',
                                size=6,
                                annotation_clip=False,
                                color=colors['text'],
                                weight=weight,
                               )
        
        for connected_block in amino_acid['connected_blocks']:
            start = connected_block[0]
            end = connected_block[-1]
    
            if clip_to_window:
                window_left, window_right = self.x_min, self.x_max
                
                if amino_acid['strand'] == '+':
                    if start > window_right or end < window_left:
                        continue
    
                    if start < window_left:
                        truncated_start = True
                        start = window_left
                    else:
                        truncated_start = False
    
                    if end > window_right:
                        truncated_end = True
                        end = window_right
                    else:
                        truncated_end = False
    
                else:
                    if end > window_right or start < window_left:
                        continue
    
                    if end < window_left:
                        truncated_end = True
                        end = window_left
                    else:
                        truncated_end = False
    
                    if start > window_right:
                        truncated_start = True
                        start = window_right
                    else:
                        truncated_start = False

            else:
                truncated_start = False
                truncated_end = False
            
            path = [
                [start + start_offset, y0],
                [start + start_offset, y1],
                [end + end_offset, y1],
                [end + end_offset, y0],
            ]
    
            if connected_block == amino_acid['connected_blocks'][0] and not truncated_start:
                path.insert(1, [start + 0.5 * start_offset, y])
    
            if connected_block == amino_acid['connected_blocks'][-1] and not truncated_end:
                path.insert(-1, [end + 1.5 * end_offset, y])

            disconnected = (len(amino_acid['connected_blocks']) > 1 or truncated_start or truncated_end)
        
            patch = plt.Polygon(path, 
                                fill=True,
                                closed=True,
                                alpha=1,
                                facecolor=colors['facecolor'],
                                edgecolor=colors['edgecolor'],
                                linewidth=0.5,
                                clip_on=False,
                                linestyle='--' if disconnected else '-',
                               )
    
            self.ax.add_patch(patch)

            self.ax.annotate(letter_to_draw,
                             xy=((start + end) / 2, y),
                             xycoords='data',
                             alpha=1,
                             ha='center',
                             va='center',
                             size=6,
                             annotation_clip=False,
                             color=colors['text'],
                             weight=weight,
                            )

    def draw_rect_in_window_coords(self, x0, x1, y0, y1, **kwargs):
        x0, x1 = sorted(map(self.variant.window_p_to_genomic_1_p, [x0, x1]))
        x0 -= 0.5
        x1 += 0.5

        if self.variant.strand == '-':
            y0 *= -1
            y1 *= -1

        self.draw_rect(x0, x1, y0, y1, **kwargs)

    def annotate_in_window_coords(self, text, x, y, **kwargs):
        x = self.variant.window_p_to_genomic_1_p(x)
    
        if self.variant.strand == '-':
            y *= -1

            if 'va' in kwargs:
                kwargs['va'] = {
                    'center': 'center',
                    'top': 'bottom',
                    'bottom': 'top',
                }[kwargs['va']]

            if 'ha' in kwargs:
                kwargs['ha'] = {
                    'center': 'center',
                    'left': 'right',
                    'right': 'left',
                }[kwargs['ha']]

            if 'xytext' in kwargs:
                x_text, y_text = kwargs['xytext']
                x_text *= -1
                y_text *= -1
                kwargs['xytext'] = (x_text, y_text)

        self.ax.annotate(text,
                            xy=(x, y),
                            xycoords='data',
                            annotation_clip=False,
                            **kwargs,
                        )

    def draw_heatmap_row(self,
                         data,
                         y,
                         height=0.6,
                         heterozygous=False,
                         min_value_to_label=0.1,
                         use_log_for_bystanders=True,
                        ):

        if use_log_for_bystanders:
            bystander_norm = matplotlib.colors.LogNorm(vmin=1e-1, vmax=100)
        else:
            bystander_norm = matplotlib.colors.Normalize(vmin=0, vmax=100)

        if self.variant.strand == '-':
            y *= -1

        if heterozygous:
            v_min = 50
        else:
            v_min = 0

        linear_norm = matplotlib.colors.Normalize(vmin=v_min, vmax=100)

        for p, value in data[data > 0.05].items():
            
            if self.x_min <= p <= self.x_max:
        
                x0 = p - 0.5
                x1 = p + 0.5
                y0 = y - height / 2
                y1 = y + height / 2

                if p == self.variant.variant_genomic_1_p:
                    cmap = effect_type_to_cmap['pathogenic', self.colorblind]
                    norm = linear_norm
                else:
                    window_p = self.variant.genomic_1_p_to_window_p(p)
                    effect = self.variant.possible_bystander_effects.get(window_p)
                    cmap = effect_to_cmap(effect, colorblind=self.colorblind)
                    norm = bystander_norm

                color = cmap(norm(value))
                
                patch = plt.Polygon([[x0, y0], [x0, y1], [x1, y1], [x1, y0]],
                                    linewidth=0,
                                    zorder=0,
                                    color=color,
                                    clip_on=False,
                                )
                self.ax.add_patch(patch)
            
                text_color = maximize_text_visibility(color)
            
                if value >= min_value_to_label:
                    self.ax.annotate(f'{value: >4.1f}',
                                        xy=(np.mean([x0, x1]), np.mean([y0, y1])),
                                        ha='center',
                                        va='center',
                                        size=4,
                                        color=text_color,
                                        annotation_clip=False,
                                        family='monospace',
                                        )

    def draw_protospacer(self, PAM, y,
                         draw_As=True,
                         label_A_positions=True,
                         draw_editing_window=True,
                         height=0.5,
                         editor_color='black',
                        ):

        p = PAM['PAM_start']

        PAM_x0 = p
        PAM_x1 = p + len(PAM['PAM']) - 1
        PAM_y0 = y - height / 2
        PAM_y1 = y + height / 2

        self.draw_rect_in_window_coords(PAM_x0,
                                   PAM_x1,
                                   PAM_y0,
                                   PAM_y1,
                       alpha=1,
                       fill=False,
                      )

        protospacer_x0 = p - len(PAM['protospacer'])
        protospacer_x1 = p - 1
        protospacer_y0 = y - height / 2
        protospacer_y1 = y + height / 2

        self.draw_rect_in_window_coords(protospacer_x0,
                       protospacer_x1,
                       protospacer_y0,
                       protospacer_y1,
                       alpha=1,
                       fill=False,
                      )

        if draw_editing_window:
            # Draw optimal editing window.

            window_height = height / 3
            window_y0 = protospacer_y1 + 0.01 * height

            optimal_offsets = PAM['editor'].deaminase.offsets_from_PAM['optimal_window']
            optimal_x0 = p + min(optimal_offsets)
            optimal_x1 = p + max(optimal_offsets)

            self.draw_rect_in_window_coords(optimal_x0,
                        optimal_x1,
                        window_y0 + window_height,
                        window_y0 + 2 * window_height,
                        alpha=0.3,
                        fill=True,
                        color='black',
                        )

            # Draw broader bystander editing window.

            bystander_offsets = PAM['editor'].deaminase.offsets_from_PAM['bystander_window']
            bystander_x0 = p + min(bystander_offsets)
            bystander_x1 = p + max(bystander_offsets)

            self.draw_rect_in_window_coords(bystander_x0,
                        bystander_x1,
                        window_y0,
                        window_y0 + window_height,
                        alpha=0.3,
                        fill=True,
                        color='black',
                        )

        if draw_As:
            self.annotate_in_window_coords('A',
                                    p + PAM['variant_offset_from_PAM'],
                                    y,
                                    color='tab:green',
                                    ha='center',
                                    va='center',
                                    fontname='DejaVu Sans',
                                    size=8,
                                    weight='bold',
                                    )

        if label_A_positions:
            variant_position_in_protospacer = PAM['variant_position_in_protospacer']
            self.annotate_in_window_coords(str(variant_position_in_protospacer),
                                      p + PAM['variant_offset_from_PAM'],
                                      y - height / 2,
                                      xytext=(0, -2),
                                      textcoords='offset points',
                                      color='tab:green',
                                      ha='center',
                                      va='top',
                                      fontname='DejaVu Sans',
                                      size=6,
                                      weight='bold',
                                     )

        for p_offset, b in enumerate(PAM['editor'].PAM):
            self.annotate_in_window_coords(b,
                                           p + p_offset,
                                           y,
                                           color='black',
                                           ha='center',
                                           va='center',
                                           fontname='DejaVu Sans',
                                           size=8,
                                           weight='normal',
                                           bbox=dict(
                                               boxstyle='square,pad=0.1',
                                               fc='white',
                                               ec='white',
                                               lw=0,
                                           ),
                                          )

        for effect in PAM['bystander_effects']:
            color = effect_to_color(effect['effect'], colorblind=self.colorblind)

            if draw_As:
                self.annotate_in_window_coords('A',
                                        p + effect['bystander_offset_from_PAM'],
                                        y,
                                        color=color,
                                        ha='center',
                                        va='center',
                                        fontname='DejaVu Sans',
                                        size=8,
                                        weight='bold',
                                        )

            if label_A_positions:
                bystander_position_in_protospacer = effect['bystander_position_in_protospacer']
                self.annotate_in_window_coords(str(bystander_position_in_protospacer),
                                        p + effect['bystander_offset_from_PAM'],
                                        y - height / 2,
                                        xytext=(0, -2),
                                        textcoords='offset points',
                                        color=color,
                                        ha='center',
                                        va='top',
                                        fontname='DejaVu Sans',
                                        size=6,
                                        weight='bold',
                                        )

        # Label editor
        self.annotate_in_window_coords(PAM['editor'].deaminase.name,
                                  PAM_x1 + 0.5,
                                  y,
                                  xytext=(3, 0),
                                  textcoords='offset points',
                                  ha='left',
                                  va='center',
                                  size=8,
                                  color=editor_color,
                                  bbox=dict(boxstyle='square,pad=0',
                                            fc='white',
                                            ec='white',
                                            lw=0,
                                           ),
                                  zorder=5,
                                 )

    def draw_legend(self, y=1):
        cumulative_offset = 0
        
        for i, (label, color) in enumerate([
            ('Effects of possible A→G edits:', 'black'),
            (f'{'installation' if self.variant.install else 'correction'} of pathogenic variant', effect_type_to_color['pathogenic', self.colorblind]),
            ('synonymous', effect_type_to_color[varcode.effects.Silent, self.colorblind]),
            ('nonsynonymous', effect_type_to_color[varcode.effects.Substitution, self.colorblind]),
            ('intronic/other', effect_type_to_color['other', self.colorblind]),
        ]):
            
            self.ax.annotate(label,
                             xy=(0, y),
                             xycoords=('axes fraction', 'data'),
                             xytext=(5, cumulative_offset),
                             textcoords='offset points',
                             ha='left',
                             va='top',
                             color=color,
                             size=8,
                            )

            cumulative_offset -= 12

    def draw_label(self, use_simple_label=False):
        if use_simple_label:
            label = self.variant.simple_label
            size = 10
            weight = 'bold'
        else:
            label = self.variant.label
            size = 8
            weight = 'normal'

        if self.variant.strand == '+':
            y = -2
            va = 'top'

        else:
            y = 3
            va = 'bottom'

        self.ax.annotate(label,
                         xy=(0.5, y),
                         xycoords=('axes fraction', 'data'),
                         color='tab:green',
                         annotation_clip=False,
                         va=va,
                         ha='center',
                         size=size,
                         weight=weight,
                        )

    def draw_genomic_coordinates(self):
        label = f'chr{self.variant.contig}\n{self.variant.variant_genomic_1_p:,}'

        if self.variant.strand == '+':
            xytext = (0, -20)
            va = 'top'

        else:
            xytext = (0, 40)
            va = 'bottom'

        self.ax.annotate(label,
                         xy=(self.variant.variant_genomic_1_p, 0),
                         xycoords='data',
                         xytext=xytext,
                         textcoords='offset points',
                         ha='center',
                         va=va,
                         annotation_clip=False,
                         size=8,
                         weight='bold',
                        )

    def draw_vertical_lines(self, y_max=10):
        y_min = -0.5

        if self.variant.strand == '-':
            y_min, y_max = -y_max, -y_min

        for x in range(self.x_min, self.x_max + 2):
            self.ax.plot([x - 0.5, x - 0.5],
                         [y_min, y_max],
                         color='black',
                         alpha=0.05,
                         clip_on=False,
                         zorder=0,
                        )

    def plot(self,
             editors,
             draw_vertical_lines=True,
             draw_label=True,
             use_simple_label=False,
             only_usable_PAMs=True,
            ):

        self.draw_sequence()

        self.draw_transcript(self.variant.MANE_transcript)

        all_PAMs = []

        for editor in editors:
            if only_usable_PAMs:
                PAMs = self.variant.usable_PAMs(editor)
            else:
                PAMs = self.variant.PAMs_with_variant_in_protospacer(editor)

            all_PAMs.extend(list(PAMs.values()))

        all_PAMs = sorted(all_PAMs, key=lambda d: d['PAM_start'])

        max_y = 0

        if self.variant.strand == '+':
            initial_gap = 3
        else:
            initial_gap = 1.5

        for protospacer_i, PAM in enumerate(all_PAMs):
            y = initial_gap + 1.5 * protospacer_i
            self.draw_protospacer(PAM, y, height=0.6)

            max_y = max(y, max_y)

        if self.variant.strand == '+':
            legend_y = -1
        else:
            legend_y = 5

        self.draw_legend(y=legend_y)

        self.ax.axis('off')

        if draw_label:
            self.draw_label(use_simple_label=use_simple_label)

        self.ax.annotate('···',
                        xy=(self.x_min, 0),
                        xycoords='data',
                        xytext=(-8, 0),
                        textcoords='offset points',
                        ha='right',
                        va='center',
                        annotation_clip=False,
                        size=18,
                       )

        self.ax.annotate('···',
                        xy=(self.x_max, 0),
                        xycoords='data',
                        xytext=(8, 0),
                        textcoords='offset points',
                        ha='left',
                        va='center',
                        annotation_clip=False,
                        size=18,
                       )

        if draw_vertical_lines:
            self.draw_vertical_lines(max_y + 1)

        self.draw_genomic_coordinates()

def extract_amino_acids(transcript):
    def connected_blocks(ps):
        blocks = []
    
        current_block = [ps.pop(0)]
    
        while ps:
            next_p = ps.pop(0)
            
            if abs(next_p - current_block[-1]) == 1:
                current_block.append(next_p)
            else:
                blocks.append(current_block)
                current_block = [next_p]
    
        blocks.append(current_block)
    
        return blocks
    
    positions = []
    
    for start_1, end_1 in transcript.coding_sequence_position_ranges:
        for p in range(start_1, end_1 + 1):
            positions.append(p)
    
    if transcript.strand == '-':
        positions = positions[::-1]
    
    aa_positions = hits.utilities.list_chunks(positions, 3)
    
    aa_sequence = Bio.Seq.translate(transcript.coding_sequence)
    
    amino_acids = []
    
    if len(aa_positions) != len(aa_sequence):
        raise ValueError
        
    for position_1, (positions, aa) in enumerate(zip(aa_positions, aa_sequence), 1):
        amino_acid = {
            'position_1': position_1,
            'strand': transcript.strand,
            'letter': aa,
            'connected_blocks': connected_blocks(positions),
        }
    
        amino_acids.append(amino_acid)

    return amino_acids
