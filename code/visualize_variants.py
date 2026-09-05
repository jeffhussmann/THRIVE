import Bio.SeqUtils
import hgvs.easy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyensembl
import varcode

import hits.genomes
import hits.utilities

ensembl = pyensembl.EnsemblRelease(species='human')

genomic_sequence_fetcher = hits.genomes.build_region_fetcher('/home/jah/indices/hg38/fasta')

MANE_transcripts = pd.read_csv('/home/jah/projects/THRIVE/data/MANE.GRCh38.v1.5.summary.txt', sep='\t')

def get_MANE_row(gene_name):
    rows = MANE_transcripts.query('symbol == @gene_name')
    
    if len(rows) != 1:
        raise ValueError(f'{gene_name} has {len(rows)} rows')
    
    MANE_row = rows.iloc[0]

    return MANE_row

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
        
        self.name = f'{self.deaminase.name} ({self.PAM})'

class ABEVariant:
    def __init__(self, var_g, MANE_row, window_on_each_side=30):
        if isinstance(var_g, str):
            var_g = hgvs.easy.parse(var_g)

        self.window_on_each_side = window_on_each_side

        self.refseq_nuc = MANE_row['RefSeq_nuc']
        self.ensembl_nuc = MANE_row['Ensembl_nuc'].split('.')[0]

        self.var_g = var_g

        self.contig = str(int(self.var_g.ac.split('.')[0].split('_')[-1]))
        if self.contig == '23':
            self.contig = 'X'

        self.variant_genomic_1_p = self.var_g.posedit.pos.start.base

        ref = self.var_g.posedit.edit.ref
        alt = self.var_g.posedit.edit.alt
        
        self.variant = varcode.Variant(self.contig, self.variant_genomic_1_p, ref, alt)

        self.effect = self.get_effect(self.variant)

        edit = str(self.var_g.posedit.edit)

        if edit == 'G>A':
            self.strand = '+'
        elif edit == 'C>T':
            self.strand = '-'
        else:
            raise ValueError(edit)

        self.other_strand = {'+': '-', '-': '+'}[self.strand]

        # TODO: handle if close to edge
        self.variant_p_in_window = self.window_on_each_side
    
        window_start = self.variant_genomic_1_p - 1 - self.window_on_each_side
        window_end = self.variant_genomic_1_p - 1 + self.window_on_each_side + 1
        window_sequence = genomic_sequence_fetcher(f'chr{self.contig}', window_start, window_end).upper()

        self.target_strand_window_sequence = window_sequence[:self.variant_p_in_window] + self.var_g.posedit.edit.alt + window_sequence[self.variant_p_in_window + 1:]

        if self.strand == '-':
            self.target_strand_window_sequence = hits.utilities.reverse_complement(self.target_strand_window_sequence)

    def get_effect(self, variant):
        effects = variant.effects()
        
        effect = effects.top_priority_effect_per_transcript_id()[self.ensembl_nuc]

        return effect

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

    def PAM(self, PAM_start, editor):
        bystander_effects = []
    
        for possible_bystander_offset in editor.deaminase.offsets_from_PAM['entire_protospacer']:
            window_p = PAM_start + possible_bystander_offset
    
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
                    'bystander_offset_from_PAM': window_p - PAM_start,
                    'bystander_position_in_protospacer': window_p - (PAM_start - 21),
                    'actual': possible_bystander_offset,
                    'window_p': window_p,
                    'genomic_1_p': self.window_p_to_genomic_1_p(window_p),
                })

        variant_offset_from_PAM = self.variant_p_in_window - PAM_start
        window_name = 'outside_protospacer'
        for possible_window_name in ['optimal_window', 'bystander_window', 'entire_protospacer']:
            if variant_offset_from_PAM in editor.deaminase.offsets_from_PAM[possible_window_name]:
                window_name = possible_window_name
                break
    
        PAM = {
            'window_name': window_name,
            'PAM_start': PAM_start,
            'protospacer': self.target_strand_window_sequence[PAM_start - 20:PAM_start],
            'PAM': self.target_strand_window_sequence[PAM_start:PAM_start + len(editor.PAM)],
            'protospacer+PAM': self.target_strand_window_sequence[PAM_start - 20:PAM_start + len(editor.PAM)],
            'bystander_effects': bystander_effects,
            'strand': self.strand,
            'variant_offset_from_PAM': self.variant_p_in_window - PAM_start,
            'variant_position_in_protospacer': self.variant_p_in_window - (PAM_start - 21),
            'editor': editor,
        }

        return PAM

    def PAMs_with_variant_in_optimal_window(self, editor):
        return {position: PAM for position, PAM in self.PAMs(editor).items() if PAM['window_name'] == 'optimal_window'}

    def PAMs_with_variant_in_bystander_window(self, editor):
        return {position: PAM for position, PAM in self.PAMs(editor).items() if PAM['window_name'] in {'optimal_window', 'bystander_window'}}

    def PAMs_with_variant_in_protospacer(self, editor):
        return {position: PAM for position, PAM in self.PAMs(editor).items() if PAM['window_name'] in {'optimal_window', 'bystander_window', 'entire_protospacer'}}

    def has_usable_PAM(self, editor):
        PAMs = self.PAMs_with_variant_in_optimal_window(editor)

        has_usable_PAM = any(PAM_is_usable(PAM) for PAM in PAMs.values())
    
        return has_usable_PAM

def effect_to_color(effect):
    if isinstance(effect, varcode.effects.Silent):
        color = 'orange'
    elif isinstance(effect, varcode.effects.Substitution):
        color = 'red'
    else:
        color = 'purple'
    return color

def PAM_is_usable(PAM):
    return not any((effect['window_name'] == 'bystander_window') and (not isinstance(effect['effect'], varcode.effects.Silent)) for effect in PAM['bystander_effects'])

class VariantDiagram:

    transcript_y = 0.8
    
    def __init__(self, variant):
        self.variant = variant

        self.fig, self.ax = plt.subplots(figsize=(10, 4))

        min_window_p = 0
        max_window_p = len(self.variant.target_strand_window_sequence) - 1
        self.x_min, self.x_max = sorted(map(self.variant.window_p_to_genomic_1_p, [min_window_p, max_window_p]))

    def draw_sequence(self):
        ax = self.ax
        
        for window_p, target_strand_b in enumerate(self.variant.target_strand_window_sequence):
            if window_p == self.variant.variant_p_in_window:
                color = 'tab:green'

                weight = 'bold'
                other_weight = 'normal'
                
            elif window_p in self.variant.possible_bystander_effects:
                effect = self.variant.possible_bystander_effects[window_p]

                color = effect_to_color(effect)

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
        
        ax.set_xlim(self.x_min, self.x_max)
        
        ax.set_ylim(-5, 5)

    def draw_rect(self, x0, x1, y0, y1,
                  alpha=1,
                  color='black',
                  fill=True,
                  clip_to_window=True,
                  line_width=1,
                  label=None,
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
            self.ax.annotate(label,
                             xy=(x0, y1),
                             xytext=(0, 3),
                             textcoords='offset points',
                             annotation_clip=False,
                             size=6, 
                             weight='bold',
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

            self.draw_rect(start - 0.5, end + 0.5, y - 0.5 * height, y + 0.5 * height,
                           0.5,
                           label=f'{gene} exon {exon_number}',
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

        letter = amino_acid['letter']
        weight = 'normal'
        
        for connected_block in amino_acid['connected_blocks']:
            position = amino_acid['position_1']

            if self.variant.variant_genomic_1_p in connected_block:
                if isinstance(self.variant.effect, varcode.effects.Substitution):
                    if self.variant.effect.aa_mutation_start_offset + 1 != position:
                        raise ValueError
                    if self.variant.effect.aa_ref != amino_acid['letter']:
                        raise ValueError

                    new_letter = self.variant.effect.aa_alt

                else:
                    new_letter = amino_acid['letter']

                colors = {
                    'facecolor': 'tab:green',
                    'edgecolor': 'black',
                    'text': 'black',
                }

                if new_letter == '':
                    new_letter = '*'

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

            self.ax.annotate(letter,
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

    def draw_protospacer(self, PAM, y,
                         draw_As=True,
                         label_A_positions=True,
                         draw_editing_window=True,
                        ):
        height = 0.5

        p = PAM['PAM_start']

        to_g_1_p = self.variant.window_p_to_genomic_1_p

        def draw_rect_in_window_coords(x0, x1, y0, y1, **kwargs):
            x0, x1 = sorted(map(to_g_1_p, [x0, x1]))
            x0 -= 0.5
            x1 += 0.5

            if self.variant.strand == '-':
                y0 *= -1
                y1 *= -1

            self.draw_rect(x0, x1, y0, y1, **kwargs)

        def annotate_in_window_coords(text, x, y, **kwargs):
            x = to_g_1_p(x)
        
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

        draw_rect_in_window_coords(p,
                       p + len(PAM['PAM']) - 1,
                       y - height / 2,
                       y + height / 2,
                       alpha=1,
                       fill=False,
                      )

        protospacer_x0 = p - len(PAM['protospacer'])
        protospacer_x1 = p - 1
        protospacer_y0 = y - height / 2
        protospacer_y1 = y + height / 2

        draw_rect_in_window_coords(protospacer_x0,
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

            draw_rect_in_window_coords(optimal_x0,
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

            draw_rect_in_window_coords(bystander_x0,
                        bystander_x1,
                        window_y0,
                        window_y0 + window_height,
                        alpha=0.3,
                        fill=True,
                        color='black',
                        )

        if draw_As:
            annotate_in_window_coords('A',
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
            annotate_in_window_coords(str(variant_position_in_protospacer),
                                    p + PAM['variant_offset_from_PAM'],
                                    y - height / 2,
                                    xytext=(0, -1),
                                    textcoords='offset points',
                                    color='tab:green',
                                    ha='center',
                                    va='top',
                                    fontname='DejaVu Sans',
                                    size=6,
                                    weight='bold',
                                    )

        for p_offset, b in enumerate(PAM['editor'].PAM):
            annotate_in_window_coords(b,
                                      p + p_offset,
                                      y,
                                      color='black',
                                      ha='center',
                                      va='center',
                                      fontname='DejaVu Sans',
                                      size=8,
                                      weight='normal',
                                     )

        for effect in PAM['bystander_effects']:
            color = effect_to_color(effect['effect'])

            if draw_As:
                annotate_in_window_coords('A',
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
                annotate_in_window_coords(str(bystander_position_in_protospacer),
                                        p + effect['bystander_offset_from_PAM'],
                                        y - height / 2,
                                        xytext=(0, -1),
                                        textcoords='offset points',
                                        color=color,
                                        ha='center',
                                        va='top',
                                        fontname='DejaVu Sans',
                                        size=6,
                                        weight='bold',
                                        )

        # Label editor
        annotate_in_window_coords(PAM['editor'].deaminase.name,
                                  protospacer_x0 - 0.5,
                                  y,
                                  xytext=(-3, 0),
                                  textcoords='offset points',
                                  ha='right',
                                  va='center',
                                  size=8,
                                 )

    def draw_legend(self):
        cumulative_offset = -5
        
        for i, (label, color) in enumerate([
            ('Effects of possible A→G edits:', 'black'),
            ('correction of pathogenic variant', 'tab:green'),
            ('synonymous', 'orange'),
            ('nonsynonymous', 'red'),
            ('intronic/other', 'purple'),
        ]):
            
            self.ax.annotate(label,
                             xy=(0, 1),
                             xycoords='axes fraction',
                             xytext=(5, cumulative_offset),
                             textcoords='offset points',
                             ha='left',
                             va='top',
                             color=color,
                             size=8,
                            )

            cumulative_offset -= 12

    def plot(self, editor_keys,
             draw_vertical_lines=True,
             use_simple_label=False,
            ):

        self.draw_sequence()

        self.draw_transcript(self.variant.MANE_transcript)

        all_PAMs = []

        for editor_key in editor_keys:
            PAMs = self.variant.PAMs_with_variant_in_protospacer(editors[editor_key])
            all_PAMs.extend(list(PAMs.values()))

        all_PAMs = sorted(all_PAMs, key=lambda d: d['PAM_start'])

        max_y = 0

        for protospacer_i, PAM in enumerate(all_PAMs):
            y = 2 + 1.75 * protospacer_i
            self.draw_protospacer(PAM, y)

            if self.variant.strand == '+':
                max_y = max(y, max_y)

        self.draw_legend()

        self.ax.axis('off')

        if use_simple_label:
            label = self.variant.simple_label
            size = 10
            weight = 'bold'
        else:
            label = self.variant.label
            size = 8
            weight = 'normal'

        self.ax.annotate(label,
                         xy=(self.variant.variant_genomic_1_p, max(3, max_y + 2)),
                         xycoords='data',
                         color='tab:green',
                         annotation_clip=False,
                         ha='center',
                         size=size,
                         weight=weight,
                        )

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

        self.ax.annotate(f'chr{self.variant.contig}\n{self.x_min:,}',
                         xy=(self.x_min, 0),
                         xycoords='data',
                         xytext=(0, -25),
                         textcoords='offset points',
                         ha='center',
                         va='top',
                         annotation_clip=False,
                         size=8,
                         weight='bold',
                        )

        if draw_vertical_lines:
            self.draw_vertical_lines()


    def draw_vertical_lines(self, y_min=-20, y_max=10):
        for x in range(self.x_min, self.x_max + 2):
            self.ax.plot([x - 0.5, x - 0.5],
                         [y_min, y_max],
                         color='black',
                         alpha=0.05,
                         clip_on=False,
                         zorder=0,
                        )

                                 
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
