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
                 batch_name,
                 var_g,
                 variant_primers,
                 positive_control_variant,
                 positive_control_primers,
                 gene,
                 heterozygous=False,
                ):

        self.base_dir = base_dir
        self.batch_name = batch_name

        self.gene = gene

        self.heterozygous = heterozygous

        sample_sheet = knock_knock.experiment.load_sample_sheet(base_dir, batch_name)

        self.sample_sheet = pd.DataFrame(sample_sheet).T
        self.sample_sheet.index.name = 'sample_name'

        self.variant = visualize_variants.BaseEditableVariant(var_g, gene)
        self.variant_primers = variant_primers
        self.positive_control_variant = positive_control_variant
        self.positive_control_primers = positive_control_primers

        def sample_name_to_PAM(sample_name):
            exp = self.exps[self.batch_name, sample_name]
            return knock_knock.effector.effectors[exp.editing_strategy.manual_effector].PAM_pattern

        self.sample_sheet['PAM'] = self.sample_sheet.index.map(sample_name_to_PAM)
        self.sample_sheet['editor'] = self.sample_sheet['condition:deaminase'] + '-' + self.sample_sheet['PAM']

    @memoized_property
    def exps(self):
        return knock_knock.experiment.get_all_experiments(self.base_dir, conditions={'batch': [self.batch_name]})

    def fractions(self, variant, relevant_primers):
        relevant_rows = self.sample_sheet.query('primers == @relevant_primers')

        exp = self.exps[self.batch_name, relevant_rows.index[0]]

        posedit = variant.var_g.posedit

        variant_p = variant.variant_genomic_1_p - exp.editing_strategy.offset_for_genomic_1(variant.var_g)

        window = hits.interval.Interval(variant_p - 10, variant_p + 10)
        precise_details = knock_knock.outcome.Details(mismatches=[knock_knock.outcome.Mismatch(variant_p, posedit.edit.ref)])

        data = {}

        for sample_name, row in relevant_rows.iterrows():
            exp = self.exps[self.batch_name, sample_name]
            outcome_fractions = knock_knock.outcome.marginalize_over_mismatches_outside_window(exp.outcome_fractions(), window)

            precise_f = outcome_fractions.get(('wild type', 'mismatches', str(precise_details)), 0)

            imprecise_f = outcome_fractions.get(('wild type', 'mismatches'), pd.Series()).sum() - precise_f

            data[sample_name] = {
                'precise': precise_f * 100,
                'imprecise': imprecise_f * 100,
                'unedited': row['condition:variant'] != str(variant.var_g)
            }

            data[sample_name].update(**row)

        df = pd.DataFrame(data).T

        df['total_modified'] = df['precise'] + df['imprecise']

        df.index.name = 'sample_name'
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
        return self.sample_sheet['condition:editor'].unique()

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
        to_plot = self.fractions(self.variant, self.variant_primers)

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
                        sem = editor_rows[quantity].sem()

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
                    
                        ax.plot([mean - sem, mean + sem], [y, y],
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
               ):

        to_plot = self.fractions(self.variant, self.variant_primers)
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

    def mismatch_fractions(self, relevant_primers):
        fs = {}

        strat = None

        relevant_rows = self.sample_sheet.query('primers == @relevant_primers')

        for sample_name, row in relevant_rows.iterrows():
            exp = self.exps[self.batch_name, sample_name]

            if exp.editing_strategy.primary_protospacer is not None:
                strat = exp.editing_strategy

            fs[exp.identifier.sample_name, row['sgRNAs'], row['condition:deaminase']] = exp.outcome_fractions()

        fs = pd.DataFrame(fs).fillna(0)
        fs.columns.names = ['sample_name', 'sgRNA', 'deaminase']

        mismatch_fractions = knock_knock.outcome.mismatch_fractions(strat,
                                                                    fs,
                                                                    relative_to_extracted_genomic_region=True,
                                                                    var_g=self.variant.var_g,
                                                                   )

        return mismatch_fractions

    def heatmap(self):
        diagram = visualize_variants.VariantDiagram(self.variant)

        diagram.plot([], draw_label=False, draw_vertical_lines=False)

        height = 0.6

        i = 2.5

        mismatch_fractions = self.mismatch_fractions(self.variant_primers)
        precise_fractions = self.fractions(self.variant, self.variant_primers)

        if self.variant.strand == '+':
            b_from = 'A'
            b_to = 'G'

        else:
            b_from = 'T'
            b_to = 'C'

        all_A_to_Gs = mismatch_fractions.xs(b_from, level='base').xs(b_to, axis=1, level='base')

        relevant_bases = [b for b in 'TCAG' if b != b_from]
        all_non_As = mismatch_fractions.loc[pd.IndexSlice[:, relevant_bases], :].droplevel('base').T.groupby('sample_name').sum().T

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

                    sample_names = list(editor_rows['sample_name'])


                    A_to_Gs = all_A_to_Gs[sample_names].mean(axis=1) * 100

                    diagram.draw_heatmap_row(A_to_Gs, y, heterozygous=self.heterozygous)

                    non_As = all_non_As[sample_names].mean(axis=1) * 100

                    diagram.draw_heatmap_row(non_As, y, heterozygous=self.heterozygous)

                    exp = self.exps[self.batch_name, sample_names[0]]

                    if not unedited:
                        editor = visualize_variants.Editor(exp.description['condition:deaminase'], exp.editing_strategy.effector.PAM_pattern)

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
                                            #alpha=0.1,
                                            clip_on=False,
                                            zorder=0.5,
                                            )
                        

                    i += height

                if diagram.variant.strand == '+':
                    label_y = sgRNA_y + height * len(editor_order)
                else:
                    label_y = sgRNA_y - height * 1.75

                if unedited:
                    sgRNA = 'unedited control'
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
