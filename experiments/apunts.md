# Apunts TFM — Fairness CNN

## Tasques prioritàries

- [x] Arreglar els problemes amb l'aleatorietat (valors diferents per cada run amb mateix `ls` i `alpha` → causa: seed no fixada per run)

## Millores pendents

- [ ] Provar altres datasets (CheXpert, etc.)
- [ ] Provar altres mètriques de fairness
- [ ] Utilitzar números primers com a llavors

## Datasets alternatius a explorar

- **CheXpert** — https://stanfordmlgroup.github.io/competitions/chexpert/
- Paper amb mateixa mètrica (TPR) aplicada a datasets diferents — https://pubmed.ncbi.nlm.nih.gov/33691020/
  - Comprovar si el nostre dataset (NIH ChestX-ray) ja apareix en el paper comparatiu de TPR
  - Explorar la dificultat d'integrar datasets addicionals
  - Aprofitar aquests papers a la memòria del TFM

## Notes tècniques

- **Variabilitat entre runs** (mateix `ls` i `alpha`): cada vegada que s'executa la seed val 1 més (seed++)






mirar altres datasets pel TFM?? com a mínim comentar-ho a la memòria (https://stanfordmlgroup.github.io/competitions/chexpert/), https://pubmed.ncbi.nlm.nih.gov/33691020/%20#%20estan%20fent%20servir%20la%20mateixa%20m%C3%A8trica%20que%20jo%20per%C3%B2%20amb%20diferents%20datasets ------------- explorar la dificutat d'afegir altres datasets

mirar si el dataset que ja utilitzem ja és utilitzat pel paper comparatiu de TPR amb varis datasets

Sobretot:
- Arreglar els problemes amb la aleatorietat

Millores:
- Provar altres datasets
- provar altres mètriques

! Aprofitar aquests papers!!

La raó per la qual apareixen valor diferents a cada run (mateix ls i alpha): per cada run --> seed++