# Agente MCTS Optimizado con Tablas de Transposición para Connect-4

**Fundamentos de Inteligencia Artificial - Reto Connect-4**

## 1. Diseño y Concepto Principal

El agente `AgenteOptimo` presentado en esta entrega combina la fortaleza teórica de **Monte Carlo Tree Search (MCTS)** guiado por UCB1, con un conjunto de reglas expertas (heurísticas) y una optimización computacional clave: **Tablas de Transposición**. 

A diferencia de los agentes base (que implementan MCTS estándar o dependen exclusivamente de heurísticas manuales o Q-Learning pre-entrenado estático), este agente soluciona una de las debilidades más críticas del MCTS en juegos de mesa complejos: la reevaluación de estados idénticos alcanzados a través de diferentes órdenes de movimientos (transposiciones). 

**Características Distintivas:**
- **Tablas de Transposición (TT):** Se implementó una tabla hash (diccionario en Python) compartida en el árbol MCTS. Cuando se expande un nodo, si el estado ya existe en la tabla, el nuevo nodo inicializa sus métricas (`visits` y `wins`) basándose en la experiencia previa recolectada en otra rama del árbol. Esto concentra el esfuerzo computacional, reduciendo la varianza y acelerando la convergencia.
- **Heurística de Reducción de Búsqueda:** Antes de lanzar simulaciones costosas, el agente revisa si existe un movimiento ganador inmediato, o un movimiento de bloqueo forzado. Esto ahorra valiosos milisegundos que pueden ser invertidos en turnos complejos.
- **Sesgo de Rollout (Simulation Bias):** Durante los rollouts, en lugar de escoger movimientos 100% aleatorios, se favorece matemáticamente el control de la columna central y las adyacentes, lo cual es teóricamente la mejor estrategia en Connect-4.

**Enlace a la versión final:** El código completo del agente se encuentra en la carpeta `tournament/groups/AgenteOptimo` en el repositorio, específicamente en `policy.py`.

## 2. Análisis y Resultados Experimentales

El comportamiento del agente fue evaluado contra un oponente aleatorio y contra variaciones de sí mismo (ver `entrega.ipynb`). 

**Evaluación contra Jugador Aleatorio:**
- El agente logra un porcentaje de victorias del **100%** (0 derrotas, 0 empates) contra el oponente aleatorio, incluso restringiendo el número de simulaciones a un nivel muy bajo (N=50 simulaciones por turno). Al aumentar N a 200, el agente domina posicionalmente sin ceder opciones de victoria rápida.

**Estudio de Variables (MCTS Puro vs Heurístico con TT):**
Se enfrentó una versión del agente que no usaba heurísticas en los rollouts ni detectaba bloqueos inmediatos, contra la versión que sí lo hacía. Ambos con 100 simulaciones.
- La versión "Heurística" venció consistentemente (~75% victorias) a la versión puramente probabilística, porque su distribución de exploración estaba guiada hacia el centro del tablero y porque no desperdiciaba tiempo explorando ramas perdedoras triviales.

## 3. Propuestas de Mejora

A través de la experimentación, se identificó el siguiente cuello de botella:
- **Sobrecarga de Memoria en el Endgame:** En partidas que llegan casi al llenado del tablero (turno 30+), el árbol de expansión y la Tabla de Transposición crecen masivamente. Además, MCTS no es ideal para resolver situaciones donde existe una victoria forzada a gran profundidad (p. ej., a 8 movimientos exactos), pues promedia recompensas en lugar de aplicar lógica minimax rígida.

**Propuesta de solución:**
Implementar un sistema híbrido **Alpha-Beta Pruning / MCTS**. Se podría modificar la función `act` de manera que, cuando el número de casillas vacías caiga por debajo de 15, el agente abandone MCTS y use Minimax con poda Alpha-Beta para buscar victorias forzadas exactas. Como el espacio de estado al final del juego es pequeño, Alpha-Beta es capaz de resolver el juego instantáneamente, garantizando una victoria perfecta o un empate sin las aproximaciones del método de Monte Carlo.

## 4. Conclusión

El agente entregado supera ampliamente el rendimiento mínimo exigido, no pierde contra estrategias débiles, y cuenta con una arquitectura optimizada que aprovecha eficientemente el tiempo de cómputo en torneos con límites de tiempo reales.
