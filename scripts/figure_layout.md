# IEEE figure sizing

All nine figure PDFs are exported at their intended print width, with **5.5-point
labels, ticks, titles, legends, and annotations**:

| Layout | PDF width | LaTeX placement |
| --- | --- | --- |
| One or two subplots | 3.5 in | `figure`, `width=\columnwidth` |
| Four subplots | 7.16 in | `figure*`, `width=\textwidth` |
| Each Figure 9 trajectory | 1.6625 in | `subfigure` of `0.475\linewidth` inside one column |

The exporter in `plot_style.py` wraps titles, fits legends below the axes, moves
boxed recording notes below the plots, and keeps the PDF width fixed. Shorter
plot areas, narrow margins, and compact legend spacing minimize vertical space.
A tight crop with a variable width would undo the
font-size consistency when LaTeX scales the figures.

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/plots/comparison.pdf}
  \caption{...}
  \label{fig:hardware-comparison}
\end{figure}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/plots/estimation_vs_measurement.pdf}
  \caption{...}
  \label{fig:model-validation}
\end{figure*}
```

## Match your exact template

IEEE template variants can have different widths. No LaTeX source is included in
this repository, so the default widths above are assumptions. To make the final
font sizes exactly equal in your template, print its dimensions in the LaTeX log:

```latex
\typeout{ANTHER column width: \the\columnwidth}
\typeout{ANTHER text width: \the\textwidth}
```

Convert the reported TeX points to inches by dividing by **72.27**, then regenerate
with those widths:

```sh
ANTHER_COLUMN_WIDTH_IN=3.5 ANTHER_TEXT_WIDTH_IN=7.16 \
  python scripts/generate_all_figures.py
```

Do not apply additional `scale`, `resizebox`, or horizontal trimming after this.

## Figure 9: two panels in one column

The two `datarate_utility_*.pdf` files are already exported at
`0.475 * ANTHER_COLUMN_WIDTH_IN`. Use your existing LaTeX arrangement unchanged:

```latex
\begin{figure}[t!]
  \centering
  \begin{subfigure}{0.475\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/plots/datarate_utility_0_1.pdf}
    \caption{$\lambda=0.10$ (Energy-prioritized)}
  \end{subfigure}\hfill
  \begin{subfigure}{0.475\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/plots/datarate_utility_0_65.pdf}
    \caption{$\lambda=0.65$ (Balanced trade-off)}
  \end{subfigure}
  \caption{...}
\end{figure}
```

Their titles are omitted because the caption/subcaptions already identify the
payload and lambda. They display cumulative energy in **kJ**, use abbreviated
in-plot reference legends, and retain the setup/reset note where applicable.
