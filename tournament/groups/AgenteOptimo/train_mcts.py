import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pickle
from groups.AgenteOptimo.policy import AgenteOptimo

def train():
    print("Iniciando pre-cálculo del árbol MCTS...")
    
    # Configuramos el agente con mucho tiempo para que explore a fondo el inicio
    agent = AgenteOptimo(max_time=20.0, num_simulations=200000)
    agent.mount()
    
    # Tablero vacío
    empty_board = np.zeros((6, 7), dtype=int)
    
    print("Explorando primeros movimientos por 20 segundos...")
    action = agent.act(empty_board)
    print(f"Acción inicial preferida tras el cálculo: {action}")
    
    print(f"Se generaron {len(agent.transposition_table)} nodos en la tabla de transposición.")
    
    # Guardar a disco
    with open(agent.model_path, 'wb') as f:
        pickle.dump(agent.transposition_table, f)
        
    print(f"Tabla guardada en {agent.model_path} de forma exitosa.")

if __name__ == "__main__":
    train()
