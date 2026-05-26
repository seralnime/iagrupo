import numpy as np
import time
import pickle
import os
from connect4.policy import Policy
from connect4.connect_state import ConnectState
#Algoritmo FirstVisitMonteCarlo Joao Alexandre Muñoz
class ProfesorFVMC(Policy):
    """
    Agente FVMC con persistencia de memoria.
    Carga valores Q pre-entrenados para no empezar desde cero.
    """
    def __init__(self, max_time=4.0, gamma=1.0, epsilon=0.1):
        super().__init__()
        self.max_time = max_time
        self.gamma = gamma       
        self.epsilon = epsilon   
        self.q_table = {}        
        self.n_table = {}        
        self.my_piece = 0
        
        # Generar ruta absoluta dinámica para evitar problemas de paths relativos
        self.model_path = os.path.join(os.path.dirname(__file__), 'q_knowledge.pkl')

    def mount(self, timeout: float = None) -> None:
        """Se ejecuta al iniciar el juego. Aquí cargamos el cerebro del agente."""
        if timeout is not None:
            self.max_time = timeout
            
        self.load_knowledge()

    def load_knowledge(self):
        """Carga la tabla Q y N desde un archivo local si existe."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.q_table = data.get('q_table', {})
                    self.n_table = data.get('n_table', {})
            except Exception as e:
                print(f"Error cargando conocimiento: {e}. Iniciando desde cero.")
        else:
            self.q_table.clear()
            self.n_table.clear()

    def save_knowledge(self):
        """Guarda la tabla Q y N en el disco duro."""
        data = {
            'q_table': self.q_table,
            'n_table': self.n_table
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)

    def act(self, s: np.ndarray) -> int:
        start_time = time.time()
        
        red_pieces = np.sum(s == -1)
        yellow_pieces = np.sum(s == 1)
        current_player = -1 if red_pieces == yellow_pieces else 1
        self.my_piece = current_player
        
        root_state = ConnectState(board=s, player=current_player)
        valid_locations = [int(c) for c in root_state.get_free_cols() if s[0, int(c)] == 0]
        
        if not valid_locations or root_state.is_final():
            return int(valid_locations[0]) if valid_locations else 0
            
        # 1. Ataque inmediato
        for a in valid_locations:
            if root_state.transition(a).get_winner() == self.my_piece:
                return a
                
        # 2. Defensa inmediata
        opp_state = ConnectState(board=s, player=-self.my_piece)
        for a in valid_locations:
            if opp_state.transition(a).get_winner() == -self.my_piece:
                return a
                
        safe_time = self.max_time * 0.9
        
        episodes = 0
        # Ahora el agente actualiza su conocimiento existente, no empieza en blanco
        while time.time() - start_time < safe_time and episodes < 150:
            self._run_fvmc_episode(root_state)
            episodes += 1
            
        return int(self._get_greedy_action(root_state, valid_locations))
        
    def _get_state_hash(self, state: ConnectState):
        return hash((state.board.tobytes(), state.player))
        
    def _get_greedy_action(self, state: ConnectState, valid_locations: list):
        s_hash = self._get_state_hash(state)
        best_q = -float('inf')
        best_a = valid_locations[0] if valid_locations else 0
        
        for a in valid_locations:
            q_val = self.q_table.get((s_hash, int(a)), 0.0)
            if q_val > best_q:
                best_q = q_val
                best_a = int(a)
        return best_a

    def _run_fvmc_episode(self, start_state: ConnectState):
        current_state = ConnectState(board=np.copy(start_state.board), player=start_state.player)
        
        if current_state.get_winner() != 0:
            return
            
        trajectory = []
        terminal = False
        reward = 0.0
        
        while not terminal:
            valid_locations = [int(c) for c in current_state.get_free_cols() if current_state.board[0, int(c)] == 0]
            if not valid_locations:
                break
                
            s_hash = self._get_state_hash(current_state)
            
            if np.random.random() < self.epsilon:
                action = int(np.random.choice(valid_locations))
            else:
                action = int(self._get_greedy_action(current_state, valid_locations))
                
            trajectory.append((s_hash, action, current_state.player))
            current_state = current_state.transition(action)
            winner = current_state.get_winner()
            
            if winner != 0:
                terminal = True
                reward = 1.0 
            elif not any(current_state.board[0, c] == 0 for c in range(7)):
                terminal = True
                reward = 0.0
        
        U = reward
        visited = set()
        
        for t in range(len(trajectory) - 1, -1, -1):
            s_hash, action, player = trajectory[t]
            state_action = (s_hash, action)
            
            if state_action not in visited:
                visited.add(state_action)
                
                if state_action not in self.n_table:
                    self.n_table[state_action] = 0
                    self.q_table[state_action] = 0.0
                    
                self.n_table[state_action] += 1
                error = U - self.q_table[state_action]
                self.q_table[state_action] += error / self.n_table[state_action]
            
            U = -self.gamma * U