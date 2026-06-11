# Guió de la Presentació — TFM Joan Massallé Richart
**Durada total:** 20 minuts  
**Tribunal:** Màster en Ciència de Dades, EPS — Universitat de Girona  
**Convocatòria:** Juny 2026

---

> **Com llegir aquest guió**
> - El text en *cursiva* és el que dius en veu alta.
> - Les notes `[→ ...]` indiquen on fer èmfasi, transicions o gestos.
> - Els temps són orientatius; si t'endarrereixes, comprimeix les diapositives 18 i 21.

---

## BLOC 1: INTRODUCCIÓ I OBJECTIUS (~5 min)

---

### Diapositiva 1 — Portada `[0:00–0:30]`

*"Bon dia. El meu nom és Joan Massallé Richart i avui us presentaré el meu Treball Final de Màster: l'anàlisi de la codificació del gènere en representacions intermèdies d'una xarxa neuronal convolucional per a l'estudi de l'equitat en predicció de pneumònia."*

`[→ Pausa breu. Deixa que el títol es llegeixi.]`

---

### Diapositiva 2 — Índex `[0:30–0:45]`

`[→ Breu. Llegeix els blocs en veu alta per orientar el tribunal.]`

*"Estructuraré la presentació en tres blocs: primer la introducció i objectius, després el desenvolupament del projecte amb els resultats, i finalment les conclusions."*

---

### Diapositiva 3 — Motivació `[0:45–2:15]`

*"En els darrers anys, el deep learning ha assolit un rendiment comparable al d'un radiòleg expert en la detecció de pneumònia a partir de radiografies de tòrax. Això és un avenç molt important. Però hi ha un problema que sovint queda amagat darrere d'aquesta xifra de rendiment global."*

`[→ Transició al punt de biaix.]`

*"Aquests models s'entrenen sobre conjunts de dades que contenen biaixos inherents — biaixos socioeconòmics, de representació, de pràctiques mèdiques. I quan un model aprèn d'aquestes dades, no sols aprèn a detectar pneumònia: aprèn i amplifica aquells biaixos. El resultat pot ser un sistema que diagnostica amb una sensibilitat diferent depenent del gènere del pacient."*

*"La pregunta natural és: com podem fer que un model sigui equitatiu entre grups demogràfics? L'estratègia més habitual és afegir una penalització d'equitat durant l'entrenament, per forçar el model a igualar el rendiment entre homes i dones."*

`[→ Transició a la hipòtesi.]`

---

### Diapositiva 4 — Problema i Hipòtesi Central `[2:15–3:45]`

*"Aquí és on apareix la paradoxa que motiva tot aquest treball."*

*"Per igualar el rendiment entre homes i dones, el model ha de ser capaç de distingir-los. I per distingir-los, ha de codificar el gènere en les seves representacions internes — tot i que el gènere mai li és donat explícitament com a entrada."*

`[→ Assenyalar el diagrama de la diapositiva.]`

*"La hipòtesi paradoxal és, per tant, que un mètode d'equitat podria incrementar la codificació de gènere en les representacions latents del model. Models dissenyats per ser equitatius que, de fet, codifiquen activament el gènere internament. Aquesta és la paradoxa de privacitat que investiguem."*

---

### Diapositiva 5 — Preguntes de Recerca `[3:45–4:45]`

*"Concretament, el treball respon tres preguntes de recerca:"*

*"La primera: es codifica informació de gènere en les representacions intermèdies d'una CNN entrenada per detectar pneumònia, malgrat que el gènere no sigui una entrada explícita?"*

*"La segona: la mida de la capa densa final — que anomenem ls — influeix en quanta informació de gènere reté el model?"*

*"I la tercera: afegir una penalització d'equitat realment incrementa la codificació interna de gènere?"*

---

### Diapositiva 6 — Objectius `[4:45–5:00]`

*"Els quatre objectius del treball són: quantificar la codificació de gènere capa a capa, caracteritzar el trade-off entre equitat i precisió diagnòstica, analitzar l'efecte de la capacitat ls, i comparar els resultats en dos conjunts de dades amb biaixos intrínsecs coneguts."*

`[→ Breu. Ja ho has dit en la motivació, no t'hi estenguis.]`

---

## BLOC 2: DESENVOLUPAMENT DEL PROJECTE (~11 min)

---

### Diapositiva 7 — Conjunts de Dades `[5:00–6:30]`

*"Hem treballat amb dos conjunts de dades públics de radiografies de tòrax."*

*"El primer és el NIH ChestX-ray, amb més de cent mil imatges. El triem perquè la literatura prèvia documenta que conté un biaix de gènere mesurable — cosa que ens permet estudiar l'efecte dels nostres mètodes sobre un biaix real i conegut."*

*"El segon és el CheXpert de Stanford, amb més de dos-cents mil imatges. També té biaix de gènere documentat, però de menor intensitat que el NIH. L'usem per contrastar si els patrons que observem al NIH es repliquen en un context diferent."*

*"Ambdós datasets es preprocessen de la mateixa manera: redimensionat a 224 per 224 píxels, conversió a RGB, normalització ImageNet, i una divisió estratificada per classe i gènere."*

---

### Diapositiva 8 — Arquitectura del Model `[6:30–8:00]`

*"L'arquitectura és una ResNet18 preentrenada sobre ImageNet, amb el backbone completament congelat. Per sobre hi afegim una capa densa intermèdia de mida configurable ls, i una capa de sortida de dues neurones per a la classificació binària."*

`[→ Assenyalar el diagrama.]`

*"Congelem el backbone intencionadament: d'aquesta manera, els únics paràmetres lliures durant l'entrenament són la capa densa i la sortida. Això ens permet aïllar l'efecte de ls i del pes de la penalització alfa sobre la codificació de gènere, sense que els canvis en el backbone contaminin l'anàlisi."*

*"Els dos paràmetres experimentals clau, doncs, són alfa — la força de la penalització d'equitat — i ls — la capacitat de la capa densa."*

---

### Diapositiva 9 — Funció de Pèrdua `[8:00–9:30]`

*"La funció de pèrdua combina la pèrdua de classificació estàndard — l'entropia creuada — amb una penalització d'equitat ponderada per alfa."*

`[→ Assenyalar la fórmula.]`

*"La penalització d'equitat és simplement el quadrat de la diferència entre la probabilitat mitjana de predicció positiva dels homes i la de les dones, calculada sobre les mostres positives del batch. Quan alpha és zero no hi ha cap restricció d'equitat. Quan alpha creix, el model rep cada vegada més pressió per igualar les seves prediccions entre grups."*

*"La mètrica de biaix que mesurem és el Gap TPR — la diferència en la taxa de vertaders positius entre homes i dones. Volem reduir aquest gap sense sacrificar l'AUC diagnòstica."*

---

### Diapositiva 10 — Linear Probing `[9:30–11:00]`

*"Per mesurar quanta informació de gènere hi ha en cada capa, usem la tècnica de linear probing."*

*"La idea és senzilla: si el model conté informació de gènere en les seves activacions, un classificador lineal entrenat sobre aquelles activacions serà capaç de predir el gènere. Fem això en sis punts de l'arquitectura: des del post-maxpool fins a la capa densa."*

`[→ Assenyalar els 6 punts.]`

*"El punt d'interès principal és la capa densa, que és la que varia en els nostres experiments. Per cada punt, entreneu una regressió logística i mesureu l'AUC ROC per a la predicció de gènere. Un AUC de 0.5 indica que no hi ha informació de gènere accessible linealment. Un AUC d'1.0 indica que el gènere es pot predir perfectament a partir d'aquelles activacions."*

---

### Diapositiva 11 — Disseny Experimental `[11:00–12:00]`

*"El disseny experimental consisteix en quatre sweeps. Els dos primers, E1 i E2, usen un règim d'alfa baix entre 0.2 i 1.2, amb valors de ls entre 4 i 48. L'objectiu és determinar si penalitzacions suaus ja tenen efecte measurable."*

*"Els dos segons, E3 i E4, usen un règim d'alfa alt entre 1 i 1000, amb valors de ls entre 2 i 8. L'objectiu és identificar el llindar crític on la penalització comença a ser efectiva."*

*"En total hem entrenat 4.240 models, cadascun repetit 20 vegades amb llavors aleatòries independents per tenir estadística robusta."*

---

### Diapositiva 12 — Resultat 1 `[12:00–13:00]`

*"El primer resultat és que la penalització és inerta per a alpha baix. En els experiments E1 i E2, el Gap TPR no varia significativament amb alfa. L'AUC de gènere a la capa densa tampoc canvia. Tota la variabilitat que observem s'explica per ls."*

*"La interpretació és clara: quan alfa és petit, la magnitud de la pèrdua d'equitat és massa petita per competir amb la pèrdua de classificació. El model simplement la ignora."*

`[→ Èmfasi: "Penalitzacions febles no serveixen. Cal alfa molt gran."]`

---

### Diapositiva 13 — Gràfica E1 `[13:00–13:15]`

*"Aquí podeu veure la figura de l'experiment E1 sobre el NIH. Cada punt és una configuració d'alpha i ls. Fixeu-vos que tots els punts formen una banda quasi vertical: les sis condicions d'alpha se superposen completament. El que diferencia els punts és ls, no alpha."*

`[→ Assenyalar el núvol de punts agrupats verticalment.]`

---

### Diapositiva 14 — Gràfica E2 `[13:15–13:30]`

*"I aquí el CheXpert. El patró és idèntic: alpha és inert. L'única diferència és que el biaix de partida és unes quatre vegades menor que al NIH."*

---

### Diapositiva 15 — Resultat 2 `[13:30–14:30]`

*"El segon resultat, i potser el més important des del punt de vista pràctic, és el trade-off entre equitat i precisió a alfa alt."*

*"Quan alfa creix per sobre d'un llindar crític — aproximadament 25 a 50 per al NIH i 10 a 25 per al CheXpert — el comportament del model canvia de forma abrupta. Per sota del llindar, l'AUC diagnòstica es manté estable i el biaix disminueix gradualment. Per sobre, el model abandona la tasca de diagnòstic i prioritza igualar les prediccions entre grups."*

`[→ Assenyalar la taula.]`

*"En números concrets per al NIH: amb alfa zero tenim un AUC diagnòstica de 0.857 i un Gap TPR de 0.05. Amb alfa 50, l'AUC cau a 0.682 i el Gap TPR baixa a 0.01. Amb alfa 1000 l'AUC s'apropa al classificador aleatori."*

*"La conclusió pràctica és que aconseguir equitat real té un cost: no hi ha forma de tenir-ho tot."*

---

### Diapositiva 16 — Gràfica E3 `[14:30–14:45]`

*"I aquí les figures dels experiments a alfa alt. El patró és molt diferent: els punts es desplacen de dalt-dreta —alta AUC, alt biaix— cap a baix-esquerra —baixa AUC, baix biaix. El llindar és visible com una transició abrupta."*

`[→ Assenyalar la diagonal descendent.]`

---

### Diapositiva 17 — Gràfica E4 `[14:45–15:00]`

*"I al CheXpert, el mateix però la transició s'inicia amb alphas més petits, consistent amb el menor biaix de partida."*

---

### Diapositiva 18 — Resum de Resultats `[15:00–15:30]`

*"Per resumir: penalitzacions febles no funcionen; existeix un llindar crític on l'equitat es compra a costa de la precisió; ls és el determinant principal de la codificació de gènere; i els patrons qualitatius es repliquen entre NIH i CheXpert, confirmant que els resultats no depenen d'un sol dataset."*

`[→ Breu. Passa ràpid a conclusions.]`

---

## BLOC 3: CONCLUSIONS (~4 min)

---

### Diapositiva 19 — Respostes a les Preguntes de Recerca `[15:30–17:00]`

*"Tornem a les tres preguntes de recerca. RQ1: sí, el gènere es codifica internament, i la font principal és el backbone preentrenat. RQ2: sí, ls més gran implica major codificació de gènere decodificable linealment. RQ3: no de forma consistent — la hipòtesi paradoxal no es confirma. La penalització no incrementa la codificació; és ls qui la determina."*

---

### Diapositiva 20 — Contribucions i Implicacions `[17:00–18:30]`

*"Les contribucions principals del treball són tres: una anàlisi sistemàtica capa a capa de la codificació de gènere sota restriccions d'equitat — que fins on sabem no s'havia fet d'aquesta manera; la identificació del llindar crític; i l'evidència que la capacitat del model domina la codificació, no la penalització."*

*"Per a un professional que vulgui usar aquests mètodes en clínica, les implicacions són clares: penalitzacions febles no serviran; les fortes requeriran sacrificar precisió diagnòstica de forma significativa; i la informació de gènere no desapareix — ve del backbone. A més, hi ha una paradoxa de privacitat: models equitatius que codifiquen activament el gènere, cosa que pot tenir implicacions legals i regulatòries."*

---

### Diapositiva 21 — Limitacions i Treball Futur `[18:30–19:30]`

*"Quant a limitacions: els biaixos intrínsecs dels datasets que hem usat són relativament baixos; caldria validar en contexts amb major disparitat. A més, el linear probing detecta només codificació lineal; el gènere podria estar codificat de forma no lineal i escapar a la nostra mesura."*

*"El treball futur passa per altres atributs protegits, mètodes d'anàlisi no lineals com MMD o informació mútua, i validació clínica amb radiòlegs."*

---

### Diapositiva 22 — Gràcies `[19:30–20:00]`

*"Moltes gràcies per la vostra atenció. Quedo a la vostra disposició per a les preguntes."*

---

## PREGUNTES FREQÜENTS DEL TRIBUNAL

A continuació, possibles preguntes que pot fer el tribunal i una guia de resposta.

---

**P: Per què heu congelat el backbone? No limita la generalització dels resultats?**

> Congelar el backbone és una decisió de disseny deliberada per aïllar l'efecte de ls i alfa. Si el backbone estigués entrenable, no podríem distingir si els canvis en la codificació de gènere provenen de la penalització o de canvis en els filtres preentrenats. La limitació és real: en aplicacions reals sovint es fa fine-tuning del backbone. El treball futur hauria de replicar l'anàlisi amb backbone entrenable.

---

**P: Per quina raó el lindar crític és diferent entre NIH i CheXpert?**

> El CheXpert té un biaix de partida molt menor (Gap TPR ≈ 0.010 vs. 0.050). Amb un biaix petit, la pèrdua d'equitat és inherentment menor en magnitud, i per tant cal un alfa menor per superar la pèrdua de classificació. El llindar depèn de la magnitud relativa de les dues pèrdues.

---

**P: Heu provat altres mètriques d'equitat a part del Gap TPR?**

> No en aquest treball. El Gap TPR — diferència en sensibilitat entre grups — és la mètrica clínicament més rellevant per a un test de screening, on els falsos negatius (diagnosticar erròniament que no hi ha pneumònia) tenen les conseqüències més greus. Altres mètriques com la paritat demogràfica o l'equitat d'oportunitats iguals podrien donar resultats diferents i és una limitació explícita del treball.

---

**P: La paradoxa de privacitat és realment una paradoxa? No és simplement una conseqüència inevitable?**

> Des d'un punt de vista tècnic, sí, és una conseqüència inevitable: un model que ha de tractar els grups de forma diferent ha d'identificar-los. El terme "paradoxa" és des del punt de vista regulatori i social: el disseny del sistema intenta ser equitatiu, però el resultat intern és que el gènere queda codificat en un model d'IA mèdica. Això pot xocar amb regulacions de privacitat com el RGPD, que restringeix l'ús d'atributs protegits.

---

**P: Per què no heu usat MIMIC-CXR?**

> L'accés a MIMIC-CXR requereix un procés de credencialització que no es va poder completar dins del termini del projecte. Hauria estat molt valuós per disposar d'un dataset amb major diversitat demogràfica.

---

**P: Creieu que el linear probing és suficient per capturar tota la codificació de gènere?**

> No. El linear probing detecta únicament la part linealment accessible del gènere en les activacions. Si el gènere és codificat de forma no lineal — per exemple, distribuït en un subespai corbat — el classificador logístic no ho detectarà. Mesurem un lower bound de la codificació de gènere. Tècniques com MMD, HSIC o informació mútua podrien detectar codificació no lineal i és la principal extensió metodològica que proposo.

---

*Fi del guió.*
