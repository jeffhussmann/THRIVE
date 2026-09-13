import hgvs.easy
import matplotlib.colors
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
                ):

        self.base_dir = base_dir
        self.batch_name = batch_name

        self.gene = gene

        sample_sheet = knock_knock.experiment.load_sample_sheet(base_dir, batch_name)

        self.sample_sheet = pd.DataFrame(sample_sheet).T
        self.sample_sheet.index.name = 'sample_name'

        self.variant = visualize_variants.ABEVariant(var_g, gene)
        self.variant_primers = variant_primers
        self.positive_control_variant = positive_control_variant
        self.positive_control_primers = positive_control_primers

        def sample_name_to_PAM(sample_name):
            exp = self.exps[self.batch_name, sample_name]
            return knock_knock.effector.effectors[exp.editing_strategy.manual_effector].PAM_pattern

        def sample_name_to_unedited(sample_name):
            exp = self.exps[self.batch_name, sample_name]
            return exp.editing_strategy.primary_protospacer is None

        self.sample_sheet['PAM'] = self.sample_sheet.index.map(sample_name_to_PAM)
        self.sample_sheet['editor'] = self.sample_sheet['condition:deaminase'] + '-' + self.sample_sheet['PAM']
        self.sample_sheet['unedited'] = self.sample_sheet.index.map(sample_name_to_unedited)

    @memoized_property
    def exps(self):
        return knock_knock.experiment.get_all_experiments(self.base_dir, conditions={'batch': [self.batch_name]})

    def fractions(self, variant, relevant_primers):
        relevant_rows = self.sample_sheet.query('primers == @relevant_primers')

        exp = self.exps[self.batch_name, relevant_rows.index[0]]

        posedit = self.variant.var_g.posedit

        variant_p = posedit.pos.start.base - (exp.editing_strategy.parameters['extracted_target_region']['target_start'] + 1)

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
            }

            data[sample_name].update(**row)

        df = pd.DataFrame(data).T

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

        return dict(zip(self.sgRNAs, marker_and_size_order))

    @memoized_property
    def editors(self):
        return self.sample_sheet['editor'].unique()

    @memoized_property
    def editor_to_color(self):
        if len(self.editors) > 10:
            palette = 'husl'
        else:
            palette = 'tab10'

        colors = sns.color_palette(palette, n_colors=len(self.editors))
        editor_to_color = dict(zip(self.editors, colors))

        return editor_to_color

    def plot(self):
        to_plot = self.fractions(self.variant, self.variant_primers)

        fig, axs = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw={'width_ratios': [2, 1]}, sharey=True)

        gap = 0.5
        y = -gap

        quantity = 'precise'

        y_ticks = []

        for sgRNA, sgRNA_rows in to_plot.groupby('sgRNAs', sort=False):
            y += gap

            sgRNA_ys = []

            sgRNA_rows.groupby('editor')[quantity].mean()

            editor_order = sgRNA_rows.groupby('editor')['precise'].mean().sort_values(ascending=False).index

            for editor in editor_order:
                sgRNA_ys.append(y)

                if sgRNA == 'AAVS1':
                    color = 'black'
                else:
                    color = self.editor_to_color[editor]

                editor_rows = sgRNA_rows.query('editor == @editor')

                for ax, quantity in zip(axs, ['precise', 'imprecise']):
            
                    mean = editor_rows[quantity].mean()
                    sem = editor_rows[quantity].sem()

                    #if quantity == 'precise':
                    #    print(f'{sgRNA}, {editor}, raw = {mean:0.2f}, of correctable = {(mean - 50) * 2:0.2f}')

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

                if sgRNA == 'AAVS1':
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

            if sgRNA != 'AAVS1':
                axs[0].annotate(sgRNA,
                                xy=(-50, np.mean(sgRNA_ys)),
                                xycoords='data',
                                xytext=(-5, 0),
                                textcoords='offset points',
                                ha='right',
                                va='center',
                                color='black',
                                size=8,
                                annotation_clip=False,
                               )

                axs[0].plot([-50, -50], [min(sgRNA_ys), max(sgRNA_ys)], 
                            '-',
                            clip_on=False,
                            color='black',
                            alpha=0.5,
                           )

            y -= 0.5

            for ax in axs:
                ax.axhline(y + gap / 2, color='black', alpha=0.5)

        axs[0].annotate('sgRNA:',
                        xy=(-50, 0),
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

        ax.set_ylim(-gap / 2, y + gap / 2)

        ax.set_yticks(y_ticks, labels=[])

        axs[0].invert_yaxis()
        axs[0].set_xlim(0, 100)
        axs[0].set_xticks([0, 25, 50, 75, 100])
        axs[1].set_xlim(0, 50)
        axs[1].set_xticks([0, 25, 50])
        axs[0].axvline(50, linestyle='--', color='black', alpha=0.3)

        axs[0].set_xlabel('% of reads with precise correction\nof pathogenic variant', size=8)
        axs[1].set_xlabel('% of reads with\nimprecise editing\n(any other mismatches)', size=8)

        return fig

    def scatter(self):
        to_plot = self.fractions(self.variant, self.variant_primers)
        fig, ax = plt.subplots(figsize=(4, 4))

        for (sgRNA, editor), rows in to_plot.groupby(['sgRNAs', 'editor']):
            x_mean = rows['precise'].mean()
            x_sem = rows['precise'].sem()

            y_mean = rows['imprecise'].mean()
            y_sem = rows['imprecise'].sem()

            if sgRNA == 'AAVS1':
                color = 'black'
            else:
                color = self.editor_to_color[editor]

            marker, size = self.sgRNA_to_marker_and_size[sgRNA]

            ax.plot([x_mean], [y_mean],
                    marker=marker,
                    markersize=size,
                    markeredgewidth=0,
                    color=color,
                    alpha=0.9,
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

        ax.set_xlim(40, 100)
        ax.set_ylim(0, 40)

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

        mismatch_fractions = knock_knock.outcome.mismatch_fractions(strat, fs, relative_to_extracted_genomic_region=True)

        return mismatch_fractions

    def heatmap(self):
        diagram = visualize_variants.VariantDiagram(self.variant)

        diagram.plot([], draw_label=False, draw_vertical_lines=False)

        height = 0.6

        i = 2.5

        mismatch_fractions = self.mismatch_fractions(self.variant_primers)

        if self.variant.strand == '+':
            b_from = 'A'
            b_to = 'G'

        else:
            b_from = 'T'
            b_to = 'C'

        all_A_to_Gs = mismatch_fractions.xs(b_from, level='base').xs(b_to, axis=1, level='base')

        relevant_bases = [b for b in 'TCAG' if b != b_from]
        all_non_As = mismatch_fractions.loc[pd.IndexSlice[:, relevant_bases], :].droplevel('base').T.groupby('sample_name').sum().T

        relevant_rows = self.sample_sheet.query('primers == @self.variant_primers')

        for sgRNA, sgRNA_rows in relevant_rows.sort_values(['unedited', 'sgRNAs'], ascending=False).groupby('sgRNAs', sort=False):
            sgRNA_y = i

            grouped = sgRNA_rows.groupby(['effector', 'condition:deaminase'])
            
            for within_sgRNA_i, (key, rows) in enumerate(grouped):
                y = i
                
                exp = self.exps[self.batch_name, rows.index[0]]

                if exp.editing_strategy.protospacer_feature is not None:
                    editor = visualize_variants.Editor(exp.description['condition:deaminase'], exp.editing_strategy.effector.PAM_pattern)

                    PAM = self.variant.PAM_from_editing_strategy(exp.editing_strategy, editor)

                    diagram.draw_protospacer(PAM,
                                        y,
                                        draw_As=False,
                                        label_A_positions=within_sgRNA_i == 0,
                                        draw_editing_window=False,
                                        height=height,
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
                    
                A_to_Gs = all_A_to_Gs[rows.index].mean(axis=1) * 100

                diagram.draw_heatmap_row(A_to_Gs, y, heterozygous=True)

                non_As = all_non_As[rows.index].mean(axis=1) * 100

                diagram.draw_heatmap_row(non_As, y, heterozygous=True)

                i += height

            if diagram.variant.strand == '+':
                label_y = sgRNA_y + height * len(grouped)
            else:
                label_y = sgRNA_y - height * 1.75

            if sgRNA == 'AAVS1':
                sgRNA = 'unedited control'
                x = diagram.variant.variant_p_in_window

            else:
                x = PAM['PAM_start'] - 10
                
            diagram.annotate_in_window_coords(
                              sgRNA,
                                x, label_y,
                                va='center',
                                ha='center',
                            )
            
            i += 1.5

        diagram.draw_vertical_lines(y_max=i)
