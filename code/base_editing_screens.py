import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import hits.interval
import hits.utilities
import knock_knock.experiment
import knock_knock.outcome

import visualize_variants

memoized_property = hits.utilities.memoized_property

class Screen:
    def __init__(self,
                 base_dir,
                 batch_names,
                 var_g,
                 variant_primers,
                 gene,
                 heterozygous=False,
                 install=False,
                 specific_unedited_variant=None,
                ):

        self.base_dir = base_dir

        if isinstance(batch_names, str):
            batch_names = [batch_names]

        self.batch_names = batch_names

        self.var_g = var_g

        self.gene = gene

        self.heterozygous = heterozygous

        self.install = install

        self.specific_unedited_variant = specific_unedited_variant

        self.variant_primers = variant_primers

    @memoized_property
    def variant(self):
        variant = visualize_variants.BaseEditableVariant(self.var_g, self.gene, install=self.install)
        return variant

    @memoized_property
    def sample_sheet(self):
        sample_sheets = {}

        for batch_name in self.batch_names:
            sample_sheet = pd.DataFrame(knock_knock.experiment.load_sample_sheet(self.base_dir, batch_name)).T
            sample_sheet.index.name = 'sample_name'

            sample_sheets[batch_name] = sample_sheet

        sample_sheet = pd.concat(sample_sheets, names=['batch_name'])

        def sample_name_to_PAM(batch_and_sample_name):
            batch_name, sample_name = batch_and_sample_name
            exp = self.exps[batch_name, sample_name]
            return knock_knock.effector.effectors[exp.editing_strategy.manual_effector].PAM_pattern

        sample_sheet['PAM'] = sample_sheet.index.map(sample_name_to_PAM)
        sample_sheet['editor'] = sample_sheet['condition:deaminase'] + '-' + sample_sheet['PAM']

        return sample_sheet

    @memoized_property
    def exps(self):
        return knock_knock.experiment.get_all_experiments(self.base_dir, conditions={'batch': self.batch_names})

    def get_exp_outcome_counts(self, batch_name, sample_name):
        exp = self.exps[batch_name, sample_name]
        return exp.outcome_counts()

    @memoized_property
    def fractions(self):
        relevant_rows = self.sample_sheet.query('primers == @self.variant_primers')

        if self.specific_unedited_variant is not None:
            relevant_rows = relevant_rows.query('`condition:variant` == @self.var_g or `condition:variant` == @self.specific_unedited_variant')

        exp = self.exps[relevant_rows.index[0]]

        posedit = self.variant.var_g.posedit

        # We want the variant in target coordinates

        mapping, strand = exp.editing_strategy.genomic_1_to_target(self.variant.var_g)

        variant_p = mapping[self.variant.variant_genomic_1_p]

        window = hits.interval.Interval(variant_p - 9, variant_p + 10)

        if self.install:
            edited_base = posedit.edit.alt
        else:
            edited_base = posedit.edit.ref

        if strand == '-':
            edited_base = hits.utilities.reverse_complement(edited_base)

        mismatch = knock_knock.outcome.Mismatch(variant_p, edited_base)
        precise_details = knock_knock.outcome.Details(mismatches=[mismatch])

        data = {}

        for (batch_name, sample_name), row in relevant_rows.iterrows():
            exp = self.exps[batch_name, sample_name]

            outcome_counts = self.get_exp_outcome_counts(batch_name, sample_name)
            total_reads = outcome_counts.sum()
            outcome_fractions = outcome_counts / total_reads
            outcome_fractions = knock_knock.outcome.marginalize_over_mismatches_outside_window(outcome_fractions, window)

            precise_f = outcome_fractions.get(('wild type', 'mismatches', str(precise_details)), 0)

            imprecise_f = outcome_fractions.get(('wild type', 'mismatches'), pd.Series()).sum() - precise_f

            contain_corrective = []

            for c, s, d in outcome_fractions.index:
                if c == 'wild type' and s == 'mismatches':
                    mismatches = knock_knock.outcome.Details.from_string(d)['mismatches'].mismatches
                    if mismatch in mismatches:
                        contain_corrective.append((c, s, d))

            contain_corrective_f = outcome_fractions.loc[contain_corrective].sum()
            
            data[batch_name, sample_name] = {
                'precise': precise_f * 100,
                'imprecise': imprecise_f * 100,
                'contains_corrective': contain_corrective_f * 100,
                'unedited': row['condition:variant'] != str(self.variant.var_g),
                'read_count': total_reads,
            }

            if self.specific_unedited_variant is not None:
                data[batch_name, sample_name]['unedited'] = row['condition:variant'] == self.specific_unedited_variant

            data[batch_name, sample_name].update(**row)

        df = pd.DataFrame(data).T

        df['total_modified'] = df['precise'] + df['imprecise']

        df.index.names = ['batch_name', 'sample_name']
        df = df.reset_index().sort_values(['unedited', 'sgRNAs'])

        return df

    @memoized_property
    def sgRNAs(self):
        return self.sample_sheet['sgRNAs'].unique()
        
    @memoized_property
    def sgRNA_to_marker_and_size(self):
        marker_and_size_order = [
            ('o', 5),
            ('s', 5),
            ('D', 5),
            ('^', 7),
        ]

        return dict(zip(self.sgRNAs, itertools.cycle(marker_and_size_order)))

    @memoized_property
    def editors(self):
        return sorted(self.sample_sheet['condition:editor'].unique())

    @memoized_property
    def editor_to_color(self):
        if len(self.editors) > 10:
            palette = 'husl'
        else:
            palette = 'tab10'

        colors = sns.color_palette(palette, n_colors=len(self.editors))
        editor_to_color = dict(zip(self.editors, colors))

        return editor_to_color

    def plot(self, plot_total_modified=False):
        to_plot = self.fractions

        num_groups = to_plot.query('unedited == False').groupby(['sgRNAs', 'condition:editor']).ngroups + 1

        if plot_total_modified:
            quantities_to_plot = [
                'total_modified',
            ]
        else:
            quantities_to_plot = [
                'precise',
                'imprecise',
            ]

        fig, axs = plt.subplots(1, len(quantities_to_plot), figsize=(3 * len(quantities_to_plot), 6 * num_groups / 25), sharey=True, squeeze=False)
        axs = axs[0]

        gap = 0.75
        y = -gap

        quantity = 'precise'

        y_ticks = []

        for unedited, unedited_rows in to_plot.groupby('unedited', sort=False):
            if unedited:
                unedited_rows = unedited_rows.copy()
                unedited_rows['sgRNAs'] = 'n/a'
                unedited_rows['condition:editor'] = 'n/a'

            for sgRNA, sgRNA_rows in unedited_rows.groupby('sgRNAs', sort=False):
                y += gap

                sgRNA_ys = []

                editor_order = sgRNA_rows.groupby('condition:editor')['precise'].mean().sort_values(ascending=False).index

                for editor in editor_order:
                    sgRNA_ys.append(y)

                    if unedited:
                        color = 'black'
                    else:
                        color = self.editor_to_color[editor]

                    editor_rows = sgRNA_rows.query('`condition:editor` == @editor')

                    for ax, quantity in zip(axs, quantities_to_plot):
                
                        mean = editor_rows[quantity].mean()
                        stds = editor_rows[quantity].std()

                        if unedited:
                            marker, size = 'o', 5
                        else:
                            marker, size = self.sgRNA_to_marker_and_size[sgRNA]
                    
                        ax.plot(mean, [y],
                                linestyle='none',
                                color=color,
                                marker=marker,
                                markeredgewidth=0,
                                markersize=size,
                                clip_on=False,
                            )
                    
                        ax.plot([mean - stds, mean + stds], [y, y],
                                linestyle='-',
                                linewidth=1.5,
                                color=color,
                                marker='none',
                                alpha=0.3,
                                clip_on=False,
                            )

                        ax.annotate(f'{mean:0.1f}',
                                    xy=(mean, y),
                                    xycoords='data',
                                    xytext=(5, 2),
                                    textcoords='offset points',
                                    ha='left',
                                    va='center',
                                    color='black',
                                    size=5,
                                    family='monospace',
                                )

                    if unedited:
                        label = 'unedited control'
                    else:
                        label = editor

                    axs[0].annotate(label,
                                xy=(0, y),
                                xycoords=('axes fraction', 'data'),
                                xytext=(-5, 0),
                                textcoords='offset points',
                                ha='right',
                                va='center',
                                color=color,
                                size=8,
                            )

                    y_ticks.append(y)

                    y += 0.5

                if not unedited:
                    axs[0].annotate(sgRNA,
                                    xy=(0, np.mean(sgRNA_ys)),
                                    xycoords='data',
                                    xytext=(-120, 0),
                                    textcoords='offset points',
                                    ha='right',
                                    va='center',
                                    color='black',
                                    size=8,
                                    annotation_clip=False,
                                )

                    #axs[0].plot([-50, -50], [min(sgRNA_ys), max(sgRNA_ys)], 
                    #            '-',
                    #            clip_on=False,
                    #            color='black',
                    #            alpha=0.5,
                    #           )

                y -= 0.5

                for ax in axs:
                    ax.axhline(y + gap / 2, color='black', alpha=0.5)

        axs[0].annotate('sgRNA:',
                        xy=(0, 0),
                        xycoords='data',
                        xytext=(-120, 20),
                        textcoords='offset points',
                        ha='right',
                        va='center',
                        color='black',
                        size=8,
                        annotation_clip=False,
                        weight='bold',
                    )

        axs[0].annotate('editor:',
                        xy=(0, 0),
                        xycoords='data',
                        xytext=(-5, 20),
                        textcoords='offset points',
                        ha='right',
                        va='center',
                        color='black',
                        size=8,
                        annotation_clip=False,
                        weight='bold',
                    )

        axs[0].set_ylim(-gap / 2, y + gap / 2)

        ax.set_yticks(y_ticks, labels=[])

        axs[0].invert_yaxis()
        axs[0].set_xlim(0, 100)
        axs[0].set_xticks([0, 25, 50, 75, 100])
        axs[-1].set_xlim(0, 100)
        axs[-1].set_xticks([0, 25, 50, 75, 100])

        if self.heterozygous:
            axs[0].axvline(50, linestyle='--', color='black', alpha=0.3)

        if plot_total_modified:
            axs[0].set_xlabel('% of reads with any modified allele', size=8)

        else:
            axs[0].set_xlabel('% of reads with precise correction\nof pathogenic variant', size=8)
            axs[-1].set_xlabel('% of reads with\nimprecise editing\n(any other mismatches)', size=8)

        return fig, axs

    def scatter(self,
                grouped=True,
                editors_to_highlight=None,
                number_markers=False,
               ):

        to_plot = self.fractions
        fig, ax = plt.subplots(figsize=(4, 4))

        for (unedited, sgRNA, editor), rows in to_plot.groupby(['unedited', 'sgRNAs', 'condition:editor']):
            if unedited:
                color = 'black'
                marker, size = 'o', 5
            else:
                color = self.editor_to_color[editor]
                marker, size = self.sgRNA_to_marker_and_size[sgRNA]

            if editors_to_highlight is None or editor in editors_to_highlight:
                alpha = 0.9

                if editors_to_highlight is not None:
                    size *= 1.5

            else:
                alpha = 0.3

            if number_markers:
                number = sgRNA.split('_')[-1]
                if number == 'NTC':
                    text = '-'
                else:
                    text = str(int(number[1:]))

                marker = rf'$\bf{{{text}}}$'

                size = 7

            if grouped:
                x_mean = rows['precise'].mean()
                x_sem = rows['precise'].sem()

                y_mean = rows['imprecise'].mean()
                y_sem = rows['imprecise'].sem()

                ax.plot([x_mean], [y_mean],
                        marker=marker,
                        markersize=size,
                        markeredgewidth=0,
                        color=color,
                        alpha=alpha,
                        clip_on=False,
                    )

                ax.plot([x_mean - x_sem, x_mean + x_sem], [y_mean, y_mean],
                        marker='none',
                        linestyle='-',
                        color=color,
                        alpha=0.3,
                    )

                ax.plot([x_mean, x_mean], [y_mean - y_sem, y_mean + y_sem],
                        marker='none',
                        linestyle='-',
                        color=color,
                        alpha=0.3,
                    )

            else:
                for _, row in rows.iterrows():
                    ax.plot([row['precise']], [row['imprecise']],
                            marker=marker,
                            markersize=size,
                            markeredgewidth=0,
                            color=color,
                            alpha=0.9,
                        )

                    ax.annotate(row['sample_name'].split('_')[-1],
                                xy=(row['precise'], row['imprecise']),
                                xytext=(0, 2),
                                textcoords='offset points',
                                ha='center',
                                va='bottom',
                                size=6,
                                color=color,
                        )


        ax.set_xlim(0)
        ax.set_ylim(0)

        if self.heterozygous:
            ax.axvline(50, linestyle='--', color='black', alpha=0.3)

        ax.set_xlabel('% of reads with precise correction\nof pathogenic variant')
        ax.set_ylabel('% of reads with imprecise editing\n(any other mismatches)')

        return fig, ax

    @memoized_property
    def mismatch_fractions(self):
        fs = {}

        strat = None

        relevant_rows = self.sample_sheet.query('primers == @self.variant_primers')

        for (batch_name, sample_name), row in relevant_rows.iterrows():
            exp = self.exps[batch_name, sample_name]

            if exp.editing_strategy.primary_protospacer is not None:
                strat = exp.editing_strategy

            outcome_counts = self.get_exp_outcome_counts(batch_name, sample_name)
            outcome_fractions = outcome_counts / outcome_counts.sum()

            fs[batch_name, sample_name, row['sgRNAs'], row['condition:deaminase']] = outcome_fractions

        fs = pd.DataFrame(fs).fillna(0)
        fs.columns.names = ['batch_name', 'sample_name', 'sgRNA', 'deaminase']

        mismatch_fractions = knock_knock.outcome.mismatch_fractions(strat,
                                                                    fs,
                                                                    relative_to_extracted_genomic_region=True,
                                                                    var_g=self.variant.var_g,
                                                                   )

        return mismatch_fractions

    def heatmap(self,
                colorblind=False,
                use_log_for_bystanders=True,
                editor_subset=None,
                sgRNA_subset=None,
                min_value_to_label=0.1,
               ):

        diagram = visualize_variants.VariantDiagram(self.variant, colorblind=colorblind)

        diagram.plot([], draw_label=False, draw_vertical_lines=False)

        height = 0.6

        i = 2.5

        mismatch_fractions = self.mismatch_fractions
        precise_fractions = self.fractions

        if self.variant.strand == '+':
            b_from = 'A'
            b_to = 'G'

        else:
            b_from = 'T'
            b_to = 'C'

        all_A_to_Gs = mismatch_fractions.xs(b_from, level='base').xs(b_to, axis=1, level='base')
        all_A_to_Gs.sort_index(axis=1, inplace=True)

        relevant_bases = [b for b in 'TCAG' if b != b_from]
        all_non_As = mismatch_fractions.loc[pd.IndexSlice[:, relevant_bases], :].droplevel('base').T.groupby(['batch_name', 'sample_name']).sum().T
        all_non_As.sort_index(axis=1, inplace=True)

        unedited_order = [
            True,
            False,
        ]

        if self.variant.strand == '-':
            unedited_order = unedited_order[::-1]

        for unedited in unedited_order:
            unedited_rows = precise_fractions.query('unedited == @unedited')

            if unedited:
                unedited_rows = unedited_rows.copy()
                unedited_rows['sgRNAs'] = 'n/a'
                unedited_rows['condition:editor'] = 'n/a'

            else:
                if editor_subset is not None:
                    unedited_rows = unedited_rows.query('`condition:editor` in @editor_subset')

                if sgRNA_subset is not None:
                    unedited_rows = unedited_rows.query('sgRNAs in @sgRNA_subset')

            sgRNA_order = sorted(unedited_rows['sgRNAs'].unique())

            if self.variant.strand == '+':
                sgRNA_order = sgRNA_order[::-1]

            for sgRNA in sgRNA_order:
                sgRNA_rows = unedited_rows.query('sgRNAs == @sgRNA')

                sgRNA_y = i

                editor_order = sgRNA_rows.groupby('condition:editor')['precise'].mean().sort_values(ascending=self.variant.strand == '+').index

                for within_sgRNA_i, editor_name in enumerate(editor_order):
                    y = i

                    editor_rows = sgRNA_rows.query('`condition:editor` == @editor_name')

                    keys = editor_rows[['batch_name', 'sample_name']].values

                    A_to_Gs = pd.concat([all_A_to_Gs[tuple(k)] for k in keys], axis=1).mean(axis=1) * 100

                    diagram.draw_heatmap_row(A_to_Gs,
                                             y,
                                             heterozygous=self.heterozygous,
                                             use_log_for_bystanders=use_log_for_bystanders,
                                             min_value_to_label=min_value_to_label,
                                            )

                    non_As = pd.concat([all_non_As[tuple(k)] for k in keys], axis=1).mean(axis=1) * 100

                    diagram.draw_heatmap_row(non_As,
                                             y,
                                             heterozygous=self.heterozygous,
                                             use_log_for_bystanders=use_log_for_bystanders,
                                             min_value_to_label=min_value_to_label,
                                            )

                    exp = self.exps[tuple(keys[0])]
                    
                    if not unedited:
                        if exp.editing_strategy.primary_protospacer is not None:
                            
                            editor = visualize_variants.Editor(editor_rows.iloc[0]['condition:deaminase'], exp.editing_strategy.effector.PAM_pattern)

                            PAM = self.variant.PAM_from_editing_strategy(exp.editing_strategy, editor)

                            diagram.draw_protospacer(PAM,
                                                    y,
                                                    draw_As=False,
                                                    label_A_positions=within_sgRNA_i == 0,
                                                    draw_editing_window=False,
                                                    height=height,
                                                    editor_color=self.editor_to_color[editor_name],
                                                    )

                    else:
                        # Unedited control

                        for line_y in [y - height / 2, y + height / 2]:
                            if self.variant.strand == '-':
                                line_y *= -1

                            diagram.ax.axhline(line_y,
                                               color='grey',
                                               clip_on=False,
                                               zorder=0.5,
                                              )
                        
                    i += height

                if diagram.variant.strand == '+':
                    label_y = sgRNA_y + height * len(editor_order)
                else:
                    label_y = sgRNA_y - height * 1.75

                    if unedited:
                        label_y += 0.5

                if unedited:
                    sgRNA = 'unedited control'
                    x = diagram.variant.variant_p_in_window

                elif exp.editing_strategy.primary_protospacer is None:
                    x = diagram.variant.variant_p_in_window

                else:
                    x = PAM['PAM_start'] - 10
                    
                diagram.annotate_in_window_coords(
                                sgRNA,
                                    x, label_y,
                                    va='center',
                                    ha='center',
                                                bbox=dict(boxstyle='square,pad=0',
                                                fc='white',
                                                ec='white',
                                                lw=0,
                                            ),
                                    zorder=5,
                                )
            
                i += 1.5

        diagram.draw_vertical_lines(y_max=i)

        return diagram
