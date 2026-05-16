# Agente MCTS con Tablas de Transposición (AgenteOptimo)

Este es el agente diseñado para el torneo de Connect-4, optimizado para tomar decisiones robustas e inteligentes combinando **Búsqueda en Árbol de Monte Carlo (MCTS)** con **Tablas de Transposición (Transposition Tables)** y heurísticas de final de juego (bloqueo y victoria inmediata).

## Características principales

- **MCTS Avanzado:** Implementa el algoritmo UCB1 para balancear la exploración y explotación.
- **Transposition Tables (Caché de estados):** Los estados evaluados durante la propagación hacia atrás (backpropagation) y expansión se guardan en una tabla hash. Esto evita que el agente recalcule las estadísticas (como `visits` y `wins`) de estados que ya ha visitado a través de una permutación de movimientos diferente. Aumenta la profundidad efectiva y la velocidad del MCTS dramáticamente.
- **Heurísticas Rápidas:** Antes de entrar en la fase MCTS pesada, evalúa si puede ganar en el siguiente turno o si debe bloquear una victoria del oponente. Durante los `rollouts` (simulaciones), da un ligero sesgo (bias) a elegir columnas centrales para maximizar sus chances de victoria.
- **Manejo de Límite de Tiempo:** Tiene un control de tiempo límite (`max_time`) que asegura que el agente juegue lo más fuerte posible dentro de las limitaciones de tiempo del torneo, interrumpiendo las simulaciones limpiamente si el tiempo está a punto de acabarse.

## Cómo ejecutar

El agente está diseñado para interactuar perfectamente con el framework de `tournament`. Al ejecutar `main.py` desde la raíz del proyecto, el archivo `tournament.py` automáticamente detectará e importará la clase `AgenteOptimo` desde la carpeta `groups/AgenteOptimo/policy.py`.

```bash
# Desde la carpeta raíz del proyecto
python tournament/main.py
```

## Estructura de archivos

- `policy.py`: Contiene toda la lógica del agente, las clases `MCTSTranspositionNode` y `AgenteOptimo`.
- `entrega.ipynb`: Notebook donde se desarrolla el análisis del desempeño y la evaluación de hiper-parámetros, incluyendo las gráficas solicitadas en la rúbrica del proyecto.
- `documento_agente.md`: Documento resumen con el diseño, análisis, propuesta de mejoras y conclusiones. Puede ser exportado a PDF para la entrega formal en MS Teams.
