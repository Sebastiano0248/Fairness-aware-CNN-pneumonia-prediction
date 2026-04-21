# Notes del TFM

## Resum TFM

Although increasingly used in medical imaging, deep learning models may reproduce biases regarding attributes such as gender or ethnicity, potentially leading to unequal diagnostic outcomes. This project examines how sensitive information is encoded into convolutional neural networks (CNNs) trained to detect pneumonia in chest X-rays and whether explicitly enforcing fairness has an impact on this encoding. 

The study employs the NIH ChestX-ray dataset for pneumonia prediction. After standard preprocessing (filtering, resizing, normalization, and train-test split), the images are used to fine-tune a pretrained model. Training uses a loss function combining the standard cross-entropy with a fairness regularization term, which penalizes differences in true positive rates between genders. Intermediate representations from five layers are extracted to analyze how gender information is encoded throughout the network.

This work aims to understand how sensitive information is encoded in the network and how the learning process, as the fairness penalization term's relative weight increases, affects this representation. We will try to improve our understanding of bias in neural models to promote more equitable diagnostic systems.

Hipòtesi: El Model té en compte el sexe per donar una resposta.

## Dubtes:

### Quin mètode de penalització s'utilitza?
Tu fairness es una penalización de diferencia de varianza en TPR (True Positive Rate) entre géneros:
```python
# src/experiment_notebook.py:56-72
return (g1_mean - g2_mean).pow(2)  # Diferencia cuadrada en probabilidades
```

### Congelar les capes de la resnet menys les últimes podria ajudar a evitar el bias de gènere? Podriem descobrir si el bias vé de la pròpia resnet18 o degut al finetunning?

- Muntar/Decidir model bàsic
- Provar aquest model decidit congelat i no congelat (exepte les últimes capes fully connected (mirar variants també)) i amb/sense la penalització de fairness

### Valdria la pena fer data augmentation? Pot ser-nos útil per evitar la diferència de loss entre gèneres?
Això ajudaria al model a entendre factors no rellevants per al diagnòstic a més d'conseguir augmentar el dataset. He pensat algunes transformacions que encaixin amb el contingut de les imatges (això provoca condicions a complir com per exemple que les imatges siguin anatòmicament plausibles):
  - Rotación leve (RandomRotation(5-10°)) — las radiografías pueden tener ligera rotación del paciente
  - Traslación leve (RandomAffine(translate=(0.05, 0.05))) — ligero descentramiento
  - Escalado leve (RandomResizedCrop con scale 0.9-1.0) — variación de distancia al tubo
  - Ajuste de brillo/contraste (ColorJitter(brightness=0.1, contrast=0.1)) — variación de exposición

Utilitzar-ho només a train, que la proporció entre homes i dones es mantingui.

### El dataset està balancejat? etc

### Valdria la pena canviar el mètode de penalització del Model?
També existeixen Lasso, Ridge i ElasticNet però aquestes penalitzacions son de complexitat, no de fairness, tot i això potser valdria la pena considerar-les a afegir per reduïr overfitting. Pel fairness existeixen altres mètodes a part de TPR que poden ser útils que convendria investigar (la última només ens és útil si estem segurs que no volem que el model sàpiga de certes features):
  - Diferencia de Odds Ecualizados (Equalized Odds): A diferencia del TPR solo, aquí penalizas la diferencia tanto en el TPR como en el FPR (Tasa de Falsos Positivos). Esto asegura que el modelo no solo acierte igual, sino que no se equivoque de más con un grupo específico.

  - Paridad Demográfica (Demographic Parity): Penaliza la diferencia en la probabilidad de que el modelo prediga la clase positiva entre grupos, independientemente de si la predicción es correcta o no.

  - Correlación de Distancia (Disentanglement): Se penaliza si las representaciones internas de la CNN (los mapas de características) están correlacionadas con la variable sensible (ej. género, raza). Si el modelo "olvida" el atributo sensible en las capas intermedias, es más difícil que sesgue el TPR.

S'estan fent proves a fariness_loss_types_test.ipynb

### Quines altres opcions a més de la regressió logística tenim per analitzar la presencia del sexe a les hidden layers? El auc que devuelve gender_predictability_study es suficiente para saber si el modelo guarda o no el sexo para determinar si se tiene o no la enfermedad?

Hi ha una sèrie d'alternatives que he de provar:

  - Linear probing amb CV                                              
    Què mesura: Separabilitat lineal del gènere                                
    Pros: Fàcil d'interpretar, AUC comparable entre capes                      
    Contres: Requereix dades suficients                                
  
  - MMD (Maximum Mean Discrepancy)                                     
    Què mesura: Distància entre distribucions M/F                           
    Pros: No requereix entrenar cap model                                      
    Contres: Sensible a hiperparàmetres (kernel)                               
  
  - Mutual Information                                                 
    Què mesura: Dependència estadística                                     
    Pros: Teòricament fonamentat                                            
    Contres: Difícil d'estimar bé en alta dimensió                          
  
  - PCA + correlació                                                
    Què mesura: Quina fracció de la variància correlaciona amb gènere          
    Pros: Interpretable, visual                                            
    Contres: Només captura relacions lineals                                   
  
  - HSIC                                                               
    Què mesura: Independència en RKHS
    Pros: El que ja estàs usant com a loss!                                    
    Contres: Costós computacionalment

  - Centered Kernel Alignment (CKA)
    Què mesura: Similitud entre representacions (p. ex. hidden layers vs. model predictor de gènere)
    Pros: Invariant a transformacions ortogonals i escalat, molt usat per comparar representacions neuronals
    Contres: No dona directament interpretació causal ni importància feature-wise

### El gradient clipping està bé pero si hi ha exploding gradients no és perquè hi hagi quelcom que està anant malament en primer lloc? o no? Quina utilitat se li dona?

Revisar si les correccions del gradient no fossin estables. Revisar si cada reentrenament dels pesos la cosa convergeix i no va estirant en direccions diferents  a cada iteració.

### Existeixen mapes d'activació com Grad-CAM que permeten veure la rellevància de cada píxel original de la imatge en el resultat final. Ho he provat a simple_training.ipynb i em dona un resultat que no acabo d'entendre.

## Dubtes logístics

- Quan entreno el model es sobreajusta. Com ho fas tu perquè no et passi?

- Sería viable canviar el dataset? Què es podria aconseguir en cas que sí?

- Volem saber si *el model* aconsegueix evitar el biax amb o sense la feature del sexe però podem alterar certes característiques de l'entrenament del model. Fins on podem alterar l'estructura del model i seguir parlant *de'l mateix model*? Un exemple és el del learning rate. Un learning rate de 1e-4 és massa, això es pot veure perquè el model es sobreajusta molt ràpidament i això es sol·luciona parcialment baixant-lo a un lr de 1e-5.

- S'ha derevisar que el dataset no tingui imatges amb una diferència en la facilitat de diagnòstic entre gèneres? Pot ser que això sigui intrínsec a la naturaleza d'aquest tipus de radiografies? Em refereixo a temes com el tamany dels pulmons, la quantitat de teixit gra/muscular i altres i com afecta això a la dificultat de diagnosticar una pneumonia.

- Les imatges son en blanc i negre, perquè les tractem com si fossin a color? Perquè el model espera imatges a color?

## Tasques:

- Reestudiar com funciona la regressió logística

- Tot i que ja estava bastant fet al meu ordinador espatllat, millorar la informació que es guarda dels entrenaments (afegir-hi un anàlisis dels resultats com la gràfica accuracy vs bias, evolució loss_train i loss_test de cada model per assegurar-me que no té overfitting, entre altres)

- Buscar mètriques de fairness

### Replicar l'experiment de'n Jerónimo i analitzar més els resultats en cerca de si apareix o no el gènere si augmenta la alpha

Té pinta que s'haura de rebaixar  la dimensió de la última capa per veure que el gènere deixa de veure's

per una capa petita final:
- sense penalització: genere no apareix
- amb penalització : genere apareix més o menys?

## Anar més enllà

- fairness_loss
- canviar les fully_connected del final
- Regressió lineal només veu combinacions lineals, i si el gènere no està codificat a la última capa de manera lineal?












# Problema 1
experiment_notebook només estratifica per class a l'hora de fer l'split, no per gènere, potser sería oportú estratificar pels dos per evitar que s'entreni amb més dades d'un gènere que de l'altre

# Problema 2
criterion no considera que hi pugui haver menys mostres de dones que homes amb patologia i per tant contribueix menys a l'aprenentatge
- Opció A — WeightedRandomSampler (preferida): repetir imatges de la dels conjunts (class x gender) minoritaris (pot causar overfitting)
- Opció B - Utilitzar el data augmentation per equiparar cada conjunt (class x gender) (pot causar overfitting però menys)
- Opció C - CE amb pesos per classe (més simple però menys precís): Calcular el pes invers de la freqüència de cada classe i passar-lo al criterion (Això ajuda amb el desequilibri de classe diagnòstica però no veu el gènere: no distingeix entre "positiu femení" i "positiu masculí"):
```
  class_counts = train_df['class'].value_counts().sort_index()
  weights = torch.tensor(1.0 / class_counts.values, dtype=torch.float).to(device)
  criterion = nn.CrossEntropyLoss(weight=weights)
```

# Problema 3
Pot existir data leakage quan no s'utilitza el data_augmentation




Setmana 08/4 - 15/04
- fer un últim zoom en el heatmap ls x alpha (pujant l'alpha y amb un ls entre 8 i 32 (?))
- Revisar si el comportament atípic de layer size = 32 amb alpha = 0.25 (assegurar-se que això no passa per aleatorietat) ---> era aleatorietat
- Millorar l'aprenentatge fent-lo acabar amb la paciència, no abans (underfitting (?))
- Que els resultats siguin mitges de varis models (alterant la llavor d'aleatorietat, per exemple)
- Mirar-se el servidor de la uni i aconseguir fer anar les gràfiques amb cuda que hauria de tenir
- Arreglar el data leakage de l'experiment
- Congelar les capes que no siguin les últimes dues (més correcte experimentalment, més ràpid d'entrenar) i veure si és realment una bona idea
- Començar la memòria, estructurar-la, explicar la situació general del projecte i el que es tenia abans que jo m'incorporés i emplenar els apartats que es puguin emplenar

Per fer:

- Buscar altres maneres de aconseguir comprovar la nostra hipòtsesi (brota el gènere a la última capa tot i penalitzar-lo, l'utilitza per t'enir-lo en compte i no fer bias (més o menys))
- Assegurar-se que donada una llavor d'aleatorietat tot l'experiment vagi exactament igual
- Mirar si el data augmentation fa millorar el model (OPCIONAL, sobretot que no ataqui el problema del fairness, útil per reforçar l'entrenament), hauria de proporcionar resultats més suaus
- Balancejar class sense data augmentation (no gènere, tranquem el nostre experiment)
- Estratificar per classe i gènere
- Arreglar el data leakage de l'experiment (ho vaig fer segur?)
- Revisar com funciona el l'split train/val/test a l'experiment (no m'ha quedat del tot clar, tenim val?)
- calcular la diferència de TPR, el gender_pred_auc i l'auc per cada ls, alpha i layer
- llegir-me ./memory/suggested_structure.md
- Revisar les possibles funcions de fairness_loss