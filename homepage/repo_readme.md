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

Data and Code from the article can be found in the [publiction_data_and_code](https://github.com/DCMLab/debussy_piano/tree/main/publication_data_and_code) folder with instructions on how to run it.

## Abstract

Claude Debussy’s personal style is typically characterised as a departure from earlier diatonic tonality, including a greater variety of pitch-class materials organised in fragmented yet coherent compositions. Exploiting the music-theoretical interpretability of Discrete Fourier Transforms over pitch-class distributions, we performed a corpus study over Debussy’s solo-piano works in order to investigate the diachronic development of such stylistic features across the composer’s lifespan. We propose quantitative heuristics for the prevalence of different pitch-class prototypes, the fragmentation of a piece across different prototypes, as well as some aspect of the overall coherence of a piece. We found strong evidence for a decrease of diatonicity in favour of octatonicity, as well as for an increase of fragmentation accompanied by non-decreasing coherence. These results contribute to the understanding of the historical development of extended-tonal harmony, while representing a fertile testing ground for the interaction of computational corpus-based methods with traditional music analytical approaches.


## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

## Naming convention

Example: `l075-01_suite_prelude`.

The filenames follow the pattern `l<###>[-##][_collection]_piece` where optional components are given in brackets `[]`.
The corresponding regular expression is `l(?P<lesure>\d{3})(?:-(?P<mvt>\d{2}))?(?:_(?P<collection>[a-z]+?))?_(?P<piece>[a-z]+)`.
The components are:

* `lesure`: A three-digit, zero-padded number corresponding to the original Lesure (1977) catalog. A mapping to the revised 2003 version can be found [on Wikipedia](https://en.wikipedia.org/wiki/List_of_compositions_by_Claude_Debussy#Piano_solo).
* `mvt` (optional): A two-digit movement number.
* `collection` (optional): If a piece is part of a collection, the first word of it's title.
* `title`: The first word of the piece's title.

Titles are given in lowercase and without diacritics.

## Overview

### debussy_childrens_corner

|         file_name          |measures|labels|standard|
|----------------------------|-------:|-----:|--------|
|l113-01_childrens_doctor    |      76|     0|        |
|l113-02_childrens_jimbos    |      81|     0|        |
|l113-03_childrens_serenade  |     124|     0|        |
|l113-04_childrens_snow      |      74|     0|        |
|l113-05_childrens_little    |      31|     0|        |
|l113-06_childrens_golliwoggs|     128|     0|        |


### debussy_deux_arabesques

|         file_name         |measures|labels|standard|
|---------------------------|-------:|-----:|--------|
|l066-01_arabesques_premiere|     107|     0|        |
|l066-02_arabesques_deuxieme|     110|     0|        |


### debussy_estampes

|       file_name        |measures|labels|standard|
|------------------------|-------:|-----:|--------|
|l100-01_estampes_pagode |      98|     0|        |
|l100-02_estampes_soiree |     136|     0|        |
|l100-03_estampes_jardins|     157|     0|        |


### debussy_etudes

|       file_name        |measures|labels|standard|
|------------------------|-------:|-----:|--------|
|l136-01_etudes_cinq     |     116|     0|        |
|l136-02_etudes_tierces  |      76|     0|        |
|l136-03_etudes_quartes  |      85|     0|        |
|l136-04_etudes_sixtes   |      59|     0|        |
|l136-05_etudes_octaves  |     121|     0|        |
|l136-06_etudes_huit     |      68|     0|        |
|l136-07_etudes_degres   |      88|     0|        |
|l136-08_etudes_agrements|      52|     0|        |
|l136-09_etudes_notes    |      84|     0|        |
|l136-10_etudes_sonorites|      75|     0|        |
|l136-11_etudes_arpeges  |      67|     0|        |
|l136-12_etudes_accords  |     181|     0|        |


### debussy_images

|       file_name        |measures|labels|standard|
|------------------------|-------:|-----:|--------|
|l087-01_images_lent     |      57|     0|        |
|l087-02_images_souvenir |      72|     0|        |
|l087-03_images_quelques |     186|     0|        |
|l110-01_images_reflets  |      95|     0|        |
|l110-02_images_hommage  |      76|     0|        |
|l110-03_images_mouvement|     177|     0|        |
|l111-01_images_cloches  |      49|     0|        |
|l111-02_images_lune     |      57|     0|        |
|l111-03_images_poissons |      97|     0|        |


### debussy_other_piano_pieces

|   file_name   |measures|labels|standard|
|---------------|-------:|-----:|--------|
|l000_etude     |      71|     0|        |
|l000_soirs     |      23|     0|        |
|l009_danse     |      92|     0|        |
|l067_mazurka   |     138|     0|        |
|l068_reverie   |     101|     0|        |
|l069_tarentelle|     333|     0|        |
|l070_ballade   |     105|     0|        |
|l071_valse     |     151|     0|        |
|l082_nocturne  |      77|     0|        |
|l099_cahier    |      55|     0|        |
|l105_masques   |     381|     0|        |
|l106_isle      |     255|     0|        |
|l108_morceau   |      27|     0|        |
|l114_petit     |      87|     0|        |
|l115_hommage   |     118|     0|        |
|l121_plus      |     148|     0|        |
|l132_berceuse  |      68|     0|        |
|l133_page      |      38|     0|        |
|l138_elegie    |      21|     0|        |


### debussy_pour_le_piano

|      file_name       |measures|labels|standard|
|----------------------|-------:|-----:|--------|
|l095-01_pour_prelude  |     163|     0|        |
|l095-02_pour_sarabande|      72|     0|        |
|l095-03_pour_toccata  |     266|     0|        |


### debussy_preludes

|         file_name          |measures|labels|standard|
|----------------------------|-------:|-----:|--------|
|l117-01_preludes_danseuses  |      31|     0|        |
|l117-02_preludes_voiles     |      64|     0|        |
|l117-03_preludes_vent       |      59|     0|        |
|l117-04_preludes_sons       |      53|     0|        |
|l117-05_preludes_collines   |      96|     0|        |
|l117-06_preludes_pas        |      36|     0|        |
|l117-07_preludes_ce         |      70|     0|        |
|l117-08_preludes_fille      |      39|     0|        |
|l117-09_preludes_serenade   |     137|     0|        |
|l117-10_preludes_cathedrale |      89|     0|        |
|l117-11_preludes_danse      |      96|     0|        |
|l117-12_preludes_minstrels  |      89|     0|        |
|l123-01_preludes_brouillards|      52|     0|        |
|l123-02_preludes_feuilles   |      52|     0|        |
|l123-03_preludes_puerta     |      90|     0|        |
|l123-04_preludes_fees       |     127|     0|        |
|l123-05_preludes_bruyeres   |      51|     0|        |
|l123-06_preludes_general    |     109|     0|        |
|l123-07_preludes_terrasse   |      45|     0|        |
|l123-08_preludes_ondine     |      74|     0|        |
|l123-09_preludes_hommage    |      54|     0|        |
|l123-10_preludes_canope     |      33|     0|        |
|l123-11_preludes_tierces    |     165|     0|        |
|l123-12_preludes_feux       |     100|     0|        |


### debussy_suite_bergamasque

|       file_name       |measures|labels|standard|
|-----------------------|-------:|-----:|--------|
|l075-01_suite_prelude  |      89|   274|2.3.0   |
|l075-02_suite_menuet   |     104|   305|2.3.0   |
|l075-03_suite_clair    |      72|   150|2.3.0   |
|l075-04_suite_passepied|     156|   284|2.3.0   |