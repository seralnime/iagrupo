# Guion y Estructura de Presentación Final - Connect 4

> **Nota para el equipo:** El tiempo será cronometrado estrictamente. Cada estudiante tiene un **máximo de 3 minutos** para su parte individual. Al terminar las 3 partes, el grupo tiene **3 minutos adicionales** para la conclusión grupal. Eviten extenderse, vayan directo al punto basándose en este guion.

---

## 1. Exposición Individual 1: AgenteOptimo (Máximo 3 minutos)

**A. Diseño del Agente (La idea principal)**
* "Mi enfoque con el AgenteOptimo fue crear una solución híbrida. Durante el inicio y medio juego utilizo **Monte Carlo Tree Search (MCTS)** con el algoritmo UCB1, apoyado por una Tabla de Transposición (`mcts_trees.pkl`) que precarga casi 15,000 nodos evaluados para no empezar a ciegas. La gran diferencia de mi agente es que, cuando quedan 14 casillas o menos, realiza una **transición dinámica a Minimax con Poda Alpha-Beta**, abandonando la probabilidad y garantizando un final matemático perfecto."

**B. Análisis y Resultados empíricos**
* "Al evaluar contra el jugador aleatorio, el win-rate es superior al 95%. Al analizar los recursos, demostré empíricamente que dándole 4.5 segundos de límite (en lugar de 1s o 2s) el MCTS logra la profundidad necesaria para evitar trampas mortales a mediano plazo."

**C. Propuestas de mejora (Identificación de debilidades)**
* "El cuello de botella está en el 'medio juego', donde el factor de ramificación explota y MCTS pierde precisión. Mi propuesta a futuro es reemplazar los rollouts aleatorios con una pequeña red neuronal (al estilo AlphaZero) para predecir el valor del tablero de inmediato, ahorrando tiempo de simulación."

---

## 2. Exposición Individual 2: Group A - ProfesorFVMC (Máximo 3 minutos)

**A. Diseño del Agente (La idea principal)**
* "Mi solución, el ProfesorFVMC, se diferencia en que implementa **First-Visit Monte Carlo (FVMC)** combinado con aprendizaje por refuerzo y persistencia de memoria. El agente guarda y carga sus experiencias pasadas en un archivo de pesos `q_knowledge.pkl`. Esto permite que el agente no empiece en cero cada partida, sino que acumule conocimiento real de sus victorias y derrotas."

**B. Análisis y Resultados empíricos**
* "Al evaluar el desempeño, el agente destruye al aleatorio aprovechando su experiencia previa. Analicé el comportamiento en función de la **tasa de exploración (épsilon)**: demostré empíricamente que un épsilon alto es útil al inicio para llenar la tabla, pero en la fase final del torneo, un enfoque *greedy* que explota el `q_knowledge.pkl` es mucho más efectivo."

**C. Propuestas de mejora (Identificación de debilidades)**
* "El principal cuello de botella es la explosión del espacio de estados. La tabla Q crece exponencialmente y buscar en memoria RAM ralentiza la toma de decisiones. La mejora propuesta es implementar una **aproximación de funciones** (como regresión lineal) que generalice características del tablero, eliminando la necesidad de guardar cada estado individualmente."

---

## 3. Exposición Individual 3: Group B - Connect4ADPAgent (Máximo 3 minutos)

**A. Diseño del Agente (La idea principal)**
* "Mi agente implementa un enfoque matemáticamente estricto: **Adaptive Dynamic Programming (ADP)**. A diferencia de los métodos de muestreo de mis compañeros, mi agente modela explícitamente el MDP, estimando probabilidades de transición y funciones de recompensa en tiempo real. Le inyecté *Reward Shaping* con recompensas sintéticas masivas al ganar o perder, para forzar a la ecuación de Bellman a propagar el valor rápidamente."

**B. Análisis y Resultados empíricos**
* "El agente nunca pierde contra el aleatorio gracias a su robustez teórica. Analizando las variables, estudié su alta sensibilidad al **factor de descuento ($\gamma$)** y al tiempo límite de la Evaluación de Políticas. Si el límite de tiempo es muy bajo, la ecuación no converge y toma decisiones subóptimas."

**C. Propuestas de mejora (Identificación de debilidades)**
* "El cuello de botella es la iteración completa sobre todo el espacio de estados $S$ conocido. En Python, estos bucles sobre miles de estados violan rápidamente los tiempos límite (4.0s). La propuesta de mejora es cambiar a **Real-Time Dynamic Programming (RTDP)**, evaluando asincrónicamente solo la vecindad local del estado actual."

---

## 4. Conclusión Grupal (Máximo 3 minutos)

*(Esta sección la expone 1 integrante en nombre de todos o se la dividen rápidamente)*

**A. El Agente Seleccionado**
* "Tras nuestro estudio comparativo, el agente seleccionado para el torneo es el **AgenteOptimo**."

**B. Justificación de la Selección y Resultados**
* "Los enfrentamos entre sí en un entorno controlado. Si bien el *ProfesorFVMC* tiene excelentes aperturas gracias a su memoria Q, y el *ADP* es teóricamente el más sólido, ambos sufren de problemas de escalabilidad en memoria (tablas gigantes) y tiempo de procesamiento."
* "Por el contrario, el **AgenteOptimo** demostró el mejor comportamiento por ser híbrido: el MCTS con la **tabla de transposición serializada precargada** domina la incertidumbre del inicio velozmente. Y una vez que quedan 14 casillas o menos, su transición a fuerza bruta (Alpha-Beta Pruning) lo vuelve invencible al asegurar victorias exactas."
* "Concluimos que su flexibilidad para adaptarse a cualquier estilo de juego (apertura por probabilidad, cierre por cálculo exacto) lo hace nuestro representante definitivo."
