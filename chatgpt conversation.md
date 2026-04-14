# Conversación ChatGPT - Bias en CNN y Fairness

> Esta es una copia de una conversación entre ChatGPT y un usuario anónimo.

---

## Usuario - Pregunta 1

Tengo una CNN que ya está entrenado y de la que estoy haciendo fine-tuning porque quiero que sepa identificar una enfermedad dadas unas radiografías. El problema es que si hago esto, el resultado de sus cálculos, que por cierto, son valores discretos, tienen un bias en función de si la persona es hombre o mujer. Estoy intentando entender por qué si yo lo penalizo… Mentira, estoy intentando entender cómo reacciona el modelo cuando lo penalizo en caso de que haya bias. Para esto estoy intentando entender el output de la penúltima capa del modelo y ver si allí existe el concepto de hombre o mujer. ¿Qué usarías para decidir si la última capa contiene esa información?

---

## ChatGPT - Respuesta 1

Buena pregunta — aquí estás entrando en terreno de **representaciones internas + fairness**, y lo que quieres hacer tiene bastante sentido: comprobar si el atributo sensible (sexo) está codificado en las features antes de la capa final.

Voy al grano: lo que quieres medir es **si el sexo es *linealmente (o fácilmente) decodificable* a partir de la penúltima capa**.

### 🔍 Idea clave: "probe" o clasificador auxiliar

Lo más estándar es usar un **linear probe**:

1. Congelas tu CNN (ya fine-tuned o no).
2. Tomas la salida de la penúltima capa (vector de features).
3. Entrenas un modelo simple (por ejemplo):
   - Regresión logística
   - SVM lineal
   - Incluso un pequeño MLP (pero mejor empezar lineal)

👉 Objetivo: predecir **hombre vs mujer** usando SOLO esas features.

### 🧠 Interpretación

- ✅ **Alta accuracy (>> 50%)**
  → El modelo **sí codifica el sexo** en esa capa.

- ❌ **Cercano a 50% (azar)**
  → No hay información útil (o no linealmente accesible).

- ⚠️ **Intermedio (60–70%)**
  → Hay señal parcial.

### 📏 Métricas importantes

No te quedes solo con accuracy:

- **AUC ROC** → más robusto si hay desbalance
- **Mutual information estimada** (más avanzado)
- **Balanced accuracy**

### 🔬 Variantes más potentes

#### 1. Probe no lineal

Entrena un pequeño MLP:
- Si el lineal falla pero el MLP funciona → la info está pero **no linealmente separable**

#### 2. CKA / similitud de representaciones

Si comparas con un modelo entrenado explícitamente para sexo:
- Usa **Centered Kernel Alignment (CKA)**
- Mide si las representaciones "se parecen"

#### 3. t-SNE / UMAP (solo exploratorio)

- Proyectas features a 2D
- Coloreas por sexo

👉 Útil visualmente, pero **no concluyente**

### ⚖️ Para tu caso (penalización de bias)

Aquí viene lo interesante:

Haz el probe **antes y después de penalizar**:

- Si la penalización funciona:
  - ↓ accuracy del probe
  - ↓ separabilidad por sexo

👉 Esto te dice si estás eliminando información sensible o solo "tapando" la salida final.

### 🧩 Alternativa más fuerte: adversarial

Si quieres algo más principled:

- Añades un **clasificador adversario** que intenta predecir el sexo
- Tu red intenta:
  - minimizar tarea principal (enfermedad)
  - maximizar error del adversario

Esto fuerza a que la representación sea **invariante al sexo**

### 🧪 Resumen claro

Lo que yo usaría como baseline:

1. Extraer features penúltima capa
2. Entrenar:
   - Regresión logística (principal)
3. Evaluar AUC
4. Comparar:
   - sin penalización
   - con penalización

👉 Si el probe funciona → hay información de sexo
👉 Si deja de funcionar → has eliminado esa señal

Si quieres, puedo ayudarte a montar el experimento en PyTorch (incluyendo cómo extraer la penúltima capa y entrenar el probe correctamente).

---

## Usuario - Pregunta 2

Interesaría no solo mirar la penúltima capa, es decir, mirar la antepenúltima, por ejemplo. Eso sería interesante de mirar. ¿Cómo puedo decidir qué modelo uso si la regresión logística, el SVM o el MLP? O sea, bajo, ¿qué debería tener en cuenta? ¿Hasta qué punto la interpretación es fiable? Si me sale una cura así del 97, estoy completamente seguro de que se representa el sexo o puede ser que haya escogido un modelo erróneo y me esté dando una cura así muy alta cuando realmente sí está el sexo. cuando realmente no está el sexo, perdón. ¿Qué intenta explicar el AUC ROC? Esto que me dices que es más robusto si hay desbalance. Estas métricas que me dices que son importantes, ¿qué explican cada una? ¿Qué problema hay que la probe sea no lineal o sea lineal? Es decir, ¿qué implicaciones tiene? ¿Cuándo debería usar CKA? ¿Y qué es esto del Umap t-SNE? Explícamelo en más detalle. ¿Qué es un clasificador adversario? ¿Y qué utilidad puede tener usarlo?

---

## ChatGPT - Respuesta 2

Muy buenas preguntas — estás tocando prácticamente todos los puntos importantes de **interpretabilidad + fairness en deep learning**. Voy a estructurarlo bien para que tengas criterios claros (no solo recetas).

### 🧠 1. ¿Mirar varias capas? (no solo la penúltima)

Sí, **es muy buena idea**.

- Capas tempranas → capturan cosas básicas (bordes, texturas)
- Capas intermedias → patrones más abstractos
- Capas finales → información más "decidida" para la tarea

👉 Si el sexo aparece:
- **temprano** → está muy ligado a la imagen (ej. anatomía)
- **tarde** → el modelo lo está usando para decidir

💡 Lo ideal:
- Hacer probes en varias capas
- Dibujar: `capa vs AUC`

### 🎯 2. ¿Qué modelo usar como probe?

La regla clave es:

> **Empieza por el modelo más simple posible**

#### 🔹 Regresión logística (baseline)

- Detecta **información lineal**
- Muy interpretable
- Menos riesgo de "inventarse" patrones

👉 Es la referencia principal

#### 🔹 SVM lineal

- Similar a logística
- A veces más robusto en high-dim

👉 Alternativa equivalente

#### 🔹 MLP (no lineal)

- Detecta **relaciones complejas**
- Más potente… pero peligroso

👉 Puede "sacar" señal muy débil

#### ⚠️ Decisión práctica

Usa este orden:

1. Regresión logística
2. Si falla → MLP pequeño
3. Compara ambos

### ⚠️ 3. ¿Puedo fiarme de un 97%?

Aquí está el punto más importante de todo.

#### ✅ Si usas un modelo SIMPLE (logístico):

→ **Sí, puedes confiar bastante**

Porque:
- No tiene capacidad de "inventar"
- Solo explota lo que ya está claramente en las features

#### ❌ Si usas un MLP grande:

→ **No puedes confiar tanto**

Porque:
- Puede sobreajustar
- Puede encontrar correlaciones débiles/no reales

#### 🔒 Para estar seguro:

Haz esto:
- Train / test split limpio
- Mejor: cross-validation
- Controla regularización

👉 Si sigue dando 97% → señal real

### 📊 4. ¿Qué mide cada métrica?

#### 🔹 Accuracy

- % de aciertos
- Problema: engañosa con desbalance

Ejemplo:
- 90% hombres → predecir siempre "hombre" = 90% accuracy

#### 🔹 AUC-ROC

**Mide: Qué tan bien separa el modelo las dos clases en general**

No depende de un threshold.

- 0.5 → azar
- 1.0 → perfecto

👉 Interpretable como:

> Probabilidad de que un positivo tenga score mayor que un negativo

💡 Por eso es mejor con desbalance

#### 🔹 Balanced accuracy

Promedia:
- accuracy en hombres
- accuracy en mujeres

👉 Evita que una clase domine

#### 🔹 Mutual Information (más avanzado)

**Mide: cuánta información sobre el sexo hay en las features**

No depende de modelo (en teoría)

### 🔍 5. Probe lineal vs no lineal

#### 🔹 Lineal

Detecta:
- información **directamente accesible**

👉 Muy importante en fairness:

> Si es lineal → el modelo puede usarlo fácilmente

#### 🔹 No lineal

Detecta:
- información **oculta / codificada**

👉 Implicación:
- Puede estar ahí
- Pero el modelo final quizá no la usa

#### 🧠 Resumen clave

| Probe | Significado |
|-------|-------------|
| Lineal funciona | El sexo está claramente codificado |
| Solo MLP funciona | Está escondido |
| Ninguno funciona | No hay señal |

### 🧬 6. ¿Qué es t-SNE / UMAP?

Son técnicas de **reducción de dimensionalidad**:

- Pasas de 512D → 2D
- Para visualizar

#### 🔹 t-SNE

- Preserva estructura local
- Bueno para clusters

#### 🔹 UMAP

- Más estable
- Preserva estructura global mejor

#### ⚠️ Problema

Son **solo visuales**:

- Puedes ver clusters "bonitos"
- Pero no significa que sean separables realmente

👉 Úsalos como intuición, no como prueba

### 🔗 7. ¿Cuándo usar CKA?

CKA mide:

> **similitud entre representaciones internas de dos modelos**

#### Útil si:

Quieres comparar:
- tu modelo vs uno entrenado para sexo
- antes vs después de penalización

👉 Si CKA alto → representaciones similares

#### Pero:

- Más complejo
- Menos directo que probes

👉 Yo lo usaría como análisis secundario

### ⚔️ 8. ¿Qué es un clasificador adversario?

Es una idea muy potente.

#### 🔧 Setup

Tienes:
- Red principal → predice enfermedad
- Red adversaria → predice sexo

#### 🎯 Objetivo

Entrenas para:
- ✅ Minimizar error de enfermedad
- ❌ Maximizar error del adversario

#### 🧠 Intuición

Obligas a la red a:

> "No codificar información sobre el sexo"

#### 💡 Resultado

- Representaciones **invariantes al sexo**
- Red más "fair"

#### 🔥 Cuándo usarlo

- Si quieres eliminar bias de verdad
- No solo medirlo
