## Članovi tima:
- Marko Šuljak, SW-35/2018

## Asistent:
- Branislav Anđelić

## Opis problema:
Zadatak projekta je da na osnovu postojećih podataka o nekretninama u dostupnom dataset-u predvidi/proceni cene nekretnina koje će iz dataset-a biti izdvojene za testiranje algoritma.
Takođe, u projektu će biti korišćeno više algoritama čiji će se rezultati potom porediti kako bi bio pronađen najefikasniji algoritam za dati problem.

## Algoritam:
Za predviđanje cene koristiće se linearna regresija, decision tree algoritam, random forest i neuralna mreža.

## Metrika:
Za metriku će biti korišćeno procentualno odstupanje od prave cene, kao i koeficijent determinacije.

## Podaci koji se koriste:
Podaci koji se koriste dostupni su na [linku](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) i predstavljaju opis nekretnina po raznim kriterijumima (kvadratura, broj soba, površina terase itd.) koji utiču na njihovu cenu.

## Validacija rešenja: 
Dataset koji koristimo biće podeljen na sledeći način:
- Trening skup - 80%
- Validacioni skup - 10%
- Test skup - 10%
