# Entrega Final Torneo Connect-4

Este repositorio contiene los archivos necesarios para la ejecución de los agentes seleccionados por el equipo para el torneo final de Connect-4.

## 1. Agentes y Clases
Se han implementado tres agentes distintos con enfoques conceptualmente diferentes. Cada uno se encuentra en su respectiva carpeta y tiene una clase propia que hereda de `Policy`:

- **Agente Optimo** (`groups/AgenteOptimo/policy.py`): Clase `AgenteOptimo`. Implementa MCTS (Monte Carlo Tree Search) con tablas de transposición y transición a Alpha-Beta Pruning en el Endgame.
- **Agente Group A** (`groups/Group A/policy.py`): Clase `ProfesorFVMC`. Implementa First-Visit Monte Carlo con tablas Q y persistencia de memoria.
- **Agente Group B** (`groups/Group B/policy.py`): Clase `Connect4ADPAgent`. Implementa Adaptive Dynamic Programming (ADP) con Reward Shaping.

## 2. Archivos Adicionales Necesarios
Para la correcta ejecución de los agentes, se han incluido los siguientes archivos y dependencias dentro de cada carpeta:

- `groups/AgenteOptimo/mcts_trees.pkl`: Árboles MCTS serializados requeridos para la ejecución óptima del AgenteOptimo, permitiendo no empezar el entrenamiento desde cero en etapas iniciales.
- `groups/Group A/q_knowledge.pkl`: Pesos y valores Q pre-entrenados para el agente ProfesorFVMC, permitiendo la persistencia del conocimiento de partidas anteriores.
- (El agente Group B no requiere archivos adicionales, ya que realiza el aprendizaje ADP de manera continua y eficiente en memoria durante el tiempo de ejecución).

## 3. Guía Breve de Ejecución
1. Asegúrese de tener las dependencias de `numpy` instaladas en el entorno (`pip install numpy`).
2. Los agentes están diseñados para ser ejecutados desde la raíz del proyecto para evitar problemas de rutas. 
3. Instancie el agente deseado importándolo de su respectivo módulo:
   ```python
   from groups.AgenteOptimo.policy import AgenteOptimo
   from groups.Group_A.policy import ProfesorFVMC
   from groups.Group_B.policy import Connect4ADPAgent
   
   # Ejemplo de inicialización
   agente = AgenteOptimo(max_time=4.5)
   agente.mount(timeout=4.5)
   ```
4. El autograder o el entorno del torneo llamará al método `act(state)` del agente, el cual devolverá el índice de la columna escogida respetando el límite de tiempo. Los paths relativos (como la lectura de `q_knowledge.pkl`) se han gestionado utilizando `os.path.dirname(__file__)` para asegurar que el agente corra correctamente sin importar de dónde sea invocado.

## 4. Presentación
La presentación final, que cubre el análisis profundo requerido y la justificación de los agentes frente a diferentes métricas, se encuentra adjunta en este paquete, junto con el documento resumen detallado.
