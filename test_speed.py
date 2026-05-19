import sys
sys.path.append(r"c:\Users\sergi\OneDrive\Documentos\universidad\9 semestre\IA\Poroyecto final\iagrupo\tournament")

import numpy as np
import time
from groups.AgenteOptimo.policy import AgenteOptimo

def test():
    policy = AgenteOptimo(num_simulations=500, rollout_depth=25, max_time=4.5)
    policy.mount(1.0)
    
    board = np.zeros((6, 7), dtype=int)
    
    start = time.time()
    for _ in range(100):
        action = policy.act(board)
    end = time.time()
    
    print(f"Time for 100 actions: {end - start:.2f} seconds")
    print(f"Time per action: {(end - start)/100:.4f} seconds")

if __name__ == "__main__":
    test()
