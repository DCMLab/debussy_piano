![Version](https://img.shields.io/github/v/release/DCMLab/debussy_piano?display_name=tag)
[![DOI](https://zenodo.org/badge/563844953.svg)](https://zenodo.org/badge/latestdoi/563844953)
![GitHub repo size](https://img.shields.io/github/repo-size/DCMLab/debussy_piano)
![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-9cf)


This is a README file for a data repository originating from the [DCML corpus initiative](https://github.com/DCMLab/dcml_corpora)
and serves as welcome page for both 

* the GitHub repo [https://github.com/DCMLab/debussy_piano](https://github.com/DCMLab/debussy_piano) and the corresponding
* documentation page [https://dcmlab.github.io/debussy_piano](https://dcmlab.github.io/debussy_piano)

For information on how to obtain and use the dataset, please refer to [this documentation page](https://dcmlab.github.io/debussy_piano/introduction).

# The Claude Debussy Solo Piano Corpus

The scores of Claude Debussy's entire oeuvre for piano solo, in uncompressed MuseScore 3 format (.mscx).

## Cite as

This dataset has been released together with the publication

> Laneve, S., Schaerf, L., Cecchetti, G., Hentschel, J., & Rohrmeier, M. (2023). The diachronic development of Debussy’s musical style: A corpus study with Discrete Fourier Transform. Humanities and Social Sciences Communications, 10(1), 289. https://doi.org/10.1057/s41599-023-01796-7

## Abstract

Claude Debussy’s personal style is typically characterised as a departure from earlier diatonic tonality, including a greater variety of pitch-class materials organised in fragmented yet coherent compositions. Exploiting the music-theoretical interpretability of Discrete Fourier Transforms over pitch-class distributions, we performed a corpus study over Debussy’s solo-piano works in order to investigate the diachronic development of such stylistic features across the composer’s lifespan. We propose quantitative heuristics for the prevalence of different pitch-class prototypes, the fragmentation of a piece across different prototypes, as well as some aspect of the overall coherence of a piece. We found strong evidence for a decrease of diatonicity in favour of octatonicity, as well as for an increase of fragmentation accompanied by non-decreasing coherence. These results contribute to the understanding of the historical development of extended-tonal harmony, while representing a fertile testing ground for the interaction of computational corpus-based methods with traditional music analytical approaches.

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

## Naming convention

The filenames follow the pattern `l<###>[-##][_collection]_piece` where optional components are given in brackets `[]`.
The corresponding regular expression is `l(?P<lesure>\d{3})(?:-(?P<mvt>\d{2}))?(?:_(?P<collection>[a-z]+?))?_(?P<piece>[a-z]+)`.
The components are:

* `lesure`: A three-digit, zero-padded number corresponding to the original Lesure (1977) catalog. A mapping to the revised 2003 version can be found [on Wikipedia](https://en.wikipedia.org/wiki/List_of_compositions_by_Claude_Debussy#Piano_solo).
* `mvt` (optional): A two-digit movement number.
* `collection` (optional): If a piece is part of a collection, the first word of it's title.
* `title`: The first word of the piece's title.

Titles are given in lowercase and without diacritics.
