import sys
import os

# Obtenemos la ruta absoluta de la carpeta actual (Group A)
current_dir = os.path.dirname(os.path.abspath(__file__))
tournament_dir = os.path.dirname(os.path.dirname(current_dir))

if tournament_dir not in sys.path:
    sys.path.insert(0, tournament_dir)

import numpy as np
from policy import ProfesorFVMC
from connect4.connect_state import ConnectState

def train_offline(episodes=10000):
    print(f"Iniciando entrenamiento offline de {episodes} episodios...")
    agent = ProfesorFVMC(epsilon=0.3) # Mayor exploración durante el entrenamiento
    
    # Intentamos cargar conocimiento previo para no borrar entrenamientos anteriores
    agent.load_knowledge()
    
    initial_board = np.zeros((6, 7), dtype=int)
    
    for i in range(episodes):
        # Estado inicial vacío
        state = ConnectState(board=initial_board, player=1)
        agent._run_fvmc_episode(state)
        
        if (i + 1) % 1000 == 0:
            print(f"Episodio {i + 1}/{episodes} completado. Estados visitados: {len(agent.q_table)}")
            
    # Guardamos los resultados
    agent.save_knowledge()
    print("Entrenamiento finalizado. Archivo 'q_knowledge.pkl' generado con éxito.")

if __name__ == "__main__":
    train_offline(episodes=300000) # Entrenar 300000 partidas simuladas