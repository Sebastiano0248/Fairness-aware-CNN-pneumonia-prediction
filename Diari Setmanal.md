# Diari Setmanal

## Setmana 11/3 - 18/3:

## Setmana 18/3 - 25/3:
Aquesta setmana he tornat a fer una repassada sencera al script original que em va passar en Jerónimo. Tot i que crec que encara no el tinc assolit al 100%, entenc bastant millor com funciona l'script algunes coses que he entès millor son:
 - Com pre-tracta les imatges
 - Com el model Resnet18 està estructurat i quines capes té a més dels retocs fets a model.py
 - Que el fine-tunning que se li aplica a la resnet18 retocada no congela capes
 - Que es fan dos càlculs del gradient durant l'entrenament, un amb la ce_loss i un altre amb la loss total (loss_ce + fairness_loss)
 - He estat redactant un arxiu Notes.md amb tots els duptes que tinc actualment i les línies de recerca que he descobert

Junt amb l'script de'n Jerónimo, faig fer un script refactoritzat amb el que he estat provant una sèrie de coses:
 - simple_training.ipynb: En aquest arxiu executo el model amb uns hiperparèmtres estandar, analitzo la seva estructura (el que he comentat abans, d'entendre-la millor) per valorar possibles millores com congelament de capes, disminució del learning_rate, augment dels batches i dotar a l'entrenament de early_stopping
 - fariness_loss_types_test.ipynb: En aquest altre notebook es busca comparar altres mètodes de penalització a més de la del TPR, el problema és que tinc una falta clara de força de còmput que, al menys de moment, no m'ha permès entrenar més de 6 o 7 models que no han donat resultats molt conclüents perquè la last_layer_size era de 256 el que li permetia la llibertat de guardar-se la feature del gènere fins al final de la NN