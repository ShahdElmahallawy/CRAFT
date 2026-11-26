Figures used by the top-level README.

The `.pdf` files are the vector originals from the paper; the `.png` files are
what the README actually displays, because GitHub does not render PDFs inline.
Regenerate the PNGs after editing a figure:

    pdftocairo -png -r 300 -singlefile fig1-query-example.pdf    fig1-query-example
    pdftocairo -png -r 600 -singlefile fig2-system-overview.pdf  fig2-system-overview

The two resolutions differ because the figures differ in physical size (468pt
vs 212pt wide); both land near 1800-1950px, which stays sharp on HiDPI screens
at the width the README renders them.
