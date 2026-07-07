# Anàlisi de la codificació del gènere en representacions intermèdies d'una CNN per estudi d'equitat en predicció de Pneumònia

TFM — Màster en Ciència de Dades, Universitat de Girona (UdG)

Estudi de com la informació de gènere queda codificada en les representacions intermèdies d'una CNN (ResNet18) entrenada per detectar pneumònia en radiografies de tòrax, i com aquesta codificació es veu afectada en aplicar una penalització de fairness durant l'entrenament. La hipòtesi central és que, paradoxalment, forçar el model a ser just l'obliga a codificar activament el gènere a les seves representacions internes. L'entrenament minimitza conjuntament l'entropia creuada i un terme de regularització que penalitza la diferència de TPR (taxa de veritables positius) entre grups. L'estudi s'ha estès posteriorment a altres atributs sensibles (edat, posició de la radiografia) i al dataset CheXpert.

---

## Datasets

| Dataset | Descripció |
|---|---|
| [NIH Chest X-Ray](https://www.kaggle.com/datasets/nih-chest-xrays/data) | 112.000 radiografies de tòrax amb etiquetes de malaltia i metadades de pacient |
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | 224.000 radiografies de Stanford amb etiquetes generades automàticament |

Abans d'utilitzar cada dataset cal executar el notebook de preprocés corresponent a `experiments/preprocesses/`.

---

## Instal·lació

```bash
git clone https://github.com/joan-massalle/Fairness-aware-CNN-pneumonia-prediction
cd Fairness-aware-CNN-pneumonia-prediction
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Principals dependències: `torch 2.10`, `torchvision 0.25`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `jupyter`.

---

## Estructura del repositori

```
├── data/                        # Datasets (no inclosos al repo)
├── experiments/
│   ├── preprocesses/            # Preprocés dels datasets (executar abans dels experiments)
│   ├── sweep_train.ipynb        # Notebook principal per llançar experiments
│   ├── sweep_analysis.ipynb     # Notebook per analitzar resultats
│   ├── fariness_loss_types_test.ipynb
│   └── ...                      # Notebooks auxiliars i còpies de sweep_train
├── src/
│   ├── experiment_notebook.py   # Lògica d'entrenament i avaluació
│   ├── fairness_losses.py       # Funcions de penalització de fairness
│   └── utils/
├── results/
│   └── sweep_bias_accuracy/
│       └── {id_experiment}/     # Resultats per experiment (timestamp)
│           ├── config.json
│           └── sweep_results.json
├── memory/                      # LaTeX de la memòria i la presentació
└── requirements.txt
```

---

## Ús

### 1. Preprocés del dataset

Executar el notebook corresponent a `experiments/preprocesses/` per generar el fitxer `experiment_data.csv` del dataset.

### 2. Llançar un experiment

Obrir `experiments/sweep_train.ipynb` i configurar els paràmetres al principi del notebook. Executar cel·la a cel·la per verificar la configuració, o llançar-lo desatès:

```bash
jupyter nbconvert --to notebook --execute --inplace experiments/sweep_train.ipynb
```

- Si `experiment = ""` → es crea un experiment nou amb ID de timestamp.
- Si `experiment = "20260620_015632"` → es reprèn un experiment existent.

### 3. Analitzar resultats

Obrir `experiments/sweep_analysis.ipynb`. Igual que sweep_train, deixar `experiment = ""` per analitzar l'últim experiment o especificar-ne un ID concret.

---

## Configuració d'un experiment

Cada experiment queda definit per un `config.json` generat automàticament:

```
{
    "N_RUNS": 3,        # Execucions per configuració (per promediar i reduir variança)
    "BASE_SEED": 42,    # Llavor base; cada run i utilitza BASE_SEED + i
    "ALPHAS": [         # Valors del pes α de la penalització de fairness
        0.1, 0.2, ..., 2.0
    ],
    "LAYER_SIZES": [    # Mides de la capa intermèdia del cap de fairness
        4, 8, ..., 48
    ],
    "MAX_EPOCHS": 100,  # Màxim d'èpoques d'entrenament
    "PATIENCE": 10,     # Early stopping: s'atura si no millora en N èpoques
    "BATCH_SIZE": 32,   # Mida del batch
    "FAIRNESS_LOSS": "tpr_parity",   # Funció de penalització de fairness
    "DATASET": "NIH_Chest_XRAY",     # Dataset utilitzat
    "IMG_SUBDIR": "sel_images"       # Subcarpeta d'imatges dins del dataset
}
```

L'experiment entrena una grid de `len(ALPHAS) × len(LAYER_SIZES)` configuracions, cadascuna `N_RUNS` vegades.

---

## Experiments executats

Cada experiment és una sweep sobre la grid α × ls amb `N_RUNS=20` (excepte els primers exploratoris).

- 20260415_000821: NIH, gènere, α (0.1-2.0), ls (4-48), N=3 - primer sweep exploratori
- 20260426_025714_old: NIH, gènere, α (0.2-1.2), ls (4-48), N=5 - versió prèvia del 20260426
- 20260426_025714: NIH, gènere, α (0.2-1.2), ls (4-48), N=20
- 20260512_035852: CheXpert, gènere, α (0.2-1.2), ls (4-48), N=20
- 20260513_195608: NIH, gènere, α (1-1000), ls (2-8), N=20 - exploració d'alphas alts
- 20260513_195713: CheXpert, gènere, α (1-1000), ls (2-8), N=20 - exploració d'alphas alts
- 20260611_195149: NIH, posició radiografia - sense config (error durant creació)
- 20260612_015945: NIH, edat (Young/Senior), α (1-1000), ls (2-8), N=20
- 20260612_015952: CheXpert, edat (Young/Senior), α (1-1000), ls (2-8), N=20
- 20260616_224807: NIH, posició radiografia, α (0.2-1.2), ls (4-48), N=20
- 20260617_201322: CheXpert, posició radiografia, α (1-1000), ls (2-8), N=20
- 20260618_125704: CheXpert, edat (Young/Senior), α (0.2-1.2), ls (4-48), N=20
- 20260618_192601: CheXpert, posició radiografia, α (0.2-1.2), ls (4-48), N=20
- 20260620_015632: CheXpert, edat (Young/Senior), α (2-10), ls (4-256), N=20 - sweep gran (ls fins a 256)
