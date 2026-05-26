# Documento Resumen de Presentación de Agentes - Connect-4

Este documento condensa los puntos clave de la presentación final y las métricas de evaluación solicitadas (según rúbrica del reto) para los tres modelos desarrollados: **AgenteOptimo**, **Group A (ProfesorFVMC)** y **Group B (Connect4ADPAgent)**.

## 1. Diseño de Agentes

Cada agente implementa una estrategia fundamentalmente distinta, demostrando versatilidad en los métodos de Inteligencia Artificial:

- **AgenteOptimo**: Utiliza Monte Carlo Tree Search (MCTS) mejorado con Tablas de Transposición. Su característica diferenciadora es la transición dinámica a Minimax con Poda Alpha-Beta (Alpha-Beta Pruning) durante el Endgame (cuando quedan pocas casillas), garantizando jugadas perfectas al final.
- **Group A (ProfesorFVMC)**: Aplica *First-Visit Monte Carlo* (FVMC) combinando aprendizaje por refuerzo y persistencia en disco (`q_knowledge.pkl`). Prioriza heurísticas de ataque y defensa inmediata, actualizando sus valores Q basándose en los episodios simulados en el tiempo restante.
- **Group B (Connect4ADPAgent)**: Implementa *Adaptive Dynamic Programming* (ADP) con *Reward Shaping*. Se diferencia al modelar explícitamente el MDP (estimando probabilidades de transición y recompensas), lo cual, complementado con recompensas sintéticas en estados terminales, permite una convergencia teórica sólida.

## 2. Análisis y Rendimiento

Se evaluaron los agentes frente a un jugador completamente aleatorio y entre sí mismos (auto-desempeño y desempeño cruzado), prestando atención a parámetros de recursos:

- **Contra Aleatorio**: Los tres agentes lograron de forma holgada el requisito de nunca perder contra el aleatorio y ganarle consistentemente (win-rate superior al 95%). El *AgenteOptimo* destrozó al jugador aleatorio aprovechando su rápida capacidad de encontrar victorias forzadas.
- **Variables de Configuración**: 
  - Para el *AgenteOptimo*, se analizó la profundidad de simulación (*rollout depth*) y el tiempo máximo por turno (evaluado a 1s, 2s y 4s). Con 4.5 segundos, la expansión del MCTS permite mirar lo suficiente a futuro para evitar trampas mortales.
  - El agente *ProfesorFVMC* demostró que el epsilon (tasa de exploración) al principio requiere ser alto, pero su desempeño mejoraba drásticamente al cargar la memoria `q_knowledge.pkl`.
  - El agente de *ADP* demostró ser altamente sensible al factor de descuento ($\gamma$) y al tamaño de los estados que lograba almacenar en su diccionario `v_hat`.

## 3. Propuesta de Mejoras y Cuellos de Botella

- **AgenteOptimo**: 
  - *Cuello de botella*: En el medio juego, el MCTS puro puede tardar en detectar bloqueos a 3 turnos vista debido a la ramificación expansiva.
  - *Mejora*: Incorporar redes neuronales ligeras para estimar el valor del nodo en vez de hacer *rollouts* aleatorios (inspirado en AlphaZero), lo cual reduciría la varianza del MCTS.
- **Group A (ProfesorFVMC)**:
  - *Cuello de botella*: El espacio de estados es inmenso y la tabla Q crece desproporcionadamente, ralentizando las búsquedas.
  - *Mejora*: Implementar aproximación de funciones (Ej. regresión lineal con características extraídas del tablero) para generalizar sobre estados no visitados.
- **Group B (Connect4ADPAgent)**:
  - *Cuello de botella*: ADP requiere iteraciones completas sobre todos los estados $S$ conocidos. Conforme avanza la partida, la evaluación de políticas viola los límites de tiempo.
  - *Mejora*: Utilizar un *Real-Time Dynamic Programming* (RTDP) enfocado solo en el estado actual y sus descendientes más probables, o reducir la evaluación de Bellman a una vecindad asincrónica local.

## 4. Comparación Grupal y Selección Final

Durante el estudio comparativo, se enfrentó a los tres agentes bajo las mismas condiciones (límite de 4.0 segundos y mismo hardware):

1. **AgenteOptimo** se consolidó como el más fuerte a corto y largo plazo gracias al empalme de MCTS con fuerza bruta (Alpha-Beta) al final de la partida. MCTS maneja el inicio con incerteza, y Alpha-Beta cierra el partido matemáticamente perfecto.
2. **ProfesorFVMC** logra grandes resultados iniciales, pero es frágil si el oponente (como AgenteOptimo) lo lleva a posiciones que no existen en su tabla Q serializada.
3. **Connect4ADPAgent** es el más elegante matemáticamente, pero en Python sufre por el rendimiento iterativo en espacios de estado masivos; ADP es mejor para ambientes más pequeños.

**Conclusión y Selección:**
Seleccionamos al **AgenteOptimo** como el representante definitivo para el torneo. Su flexibilidad de recursos, resistencia contra diferentes oponentes y el uso de los **árboles MCTS serializados** lo hacen la opción más robusta y competitiva, destacando un desempeño superior y sostenido.
