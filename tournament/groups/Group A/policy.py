import numpy as np
import time
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class FVMCAgent(Policy):
    """
    Agente riguroso basado en Iteración General de Políticas (GPI) 
    con First-Visit Monte Carlo (FVMC), explorando con epsilon-greedy 
    y aprendiendo bajo un modelo de Juego de Markov Alternado de suma cero.
    """
    def __init__(self, max_time=4.0, gamma=1.0, epsilon=0.1):
        super().__init__()
        self.max_time = max_time
        self.gamma = gamma       # Factor de descuento para el valor de la utilidad[cite: 9]
        self.epsilon = epsilon   # Probabilidad de exploración para estrategia epsilon-greedy[cite: 8]
        self.q_table = {}        # Función Q única compartida
        self.n_table = {}        # Contadores de estabilización
        self.my_piece = 0

    def mount(self, timeout: float = None) -> None:
        self.q_table.clear()
        self.n_table.clear()
        if timeout is not None:
            self.max_time = timeout

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
            
        # Heurística de Lookahead (1-step): Verificaciones inmediatas de supervivencia[cite: 8]
        # 1. Ataque: Si podemos ganar inmediatamente, tomamos la acción sin dudarlo.
        for a in valid_locations:
            if root_state.transition(a).get_winner() == self.my_piece:
                return a
                
        # 2. Defensa: Si el oponente tiene una victoria inmediata en la siguiente jugada, la bloqueamos.
        opp_state = ConnectState(board=s, player=-self.my_piece)
        for a in valid_locations:
            if opp_state.transition(a).get_winner() == -self.my_piece:
                return a
                
        # Imposición del límite estricto de tiempo: 
        # Utiliza el margen de seguridad de max_time.
        safe_time = self.max_time * 0.9
        
        # Generación iterativa de episodios (Trials) limitando tanto en tiempo como 
        # en número máximo de iteraciones (150) para evitar agotar los 600s de Gradescope.
        episodes = 0
        while time.time() - start_time < safe_time and episodes < 150:
            self._run_fvmc_episode(root_state)
            episodes += 1
            
        best_action = self._get_greedy_action(root_state, valid_locations)
        return int(best_action)
        
    def _get_state_hash(self, state: ConnectState):
        """Hashea el estado matricial de forma inmutable."""
        return hash((state.board.tobytes(), state.player))
        
    def _get_greedy_action(self, state: ConnectState, valid_locations: list):
        """Retorna arg max de \hat{q}_t(a) evaluando la tabla Q desde la perspectiva del jugador actual[cite: 8]."""
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
        """
        Genera una trayectoria conceptual manteniendo la pureza estricta 
        de la evaluación First-Visit Monte Carlo[cite: 9].
        """
        current_state = ConnectState(board=np.copy(start_state.board), player=start_state.player)
        
        if current_state.get_winner() != 0:
            return
            
        trajectory = []
        terminal = False
        reward = 0.0
        
        # 1. Generar la trayectoria estocástica (Trial \tau)[cite: 9]
        while not terminal:
            valid_locations = [int(c) for c in current_state.get_free_cols() if current_state.board[0, int(c)] == 0]
            if not valid_locations:
                break
                
            s_hash = self._get_state_hash(current_state)
            
            # Equilibrio de exploración vs. explotación[cite: 8]
            if np.random.random() < self.epsilon:
                action = int(np.random.choice(valid_locations))
            else:
                action = int(self._get_greedy_action(current_state, valid_locations))
                
            # Registramos quién hizo el movimiento antes de transicionar
            trajectory.append((s_hash, action, current_state.player))
            
            current_state = current_state.transition(action)
            winner = current_state.get_winner()
            
            if winner != 0:
                terminal = True
                # La recompensa de un triunfo es lógicamente 1.0 para el jugador que ejecutó la última acción
                reward = 1.0 
            elif not any(current_state.board[0, c] == 0 for c in range(7)):
                terminal = True
                reward = 0.0
        
        # 2. Evaluación de Políticas mediante First-Visit Monte Carlo[cite: 9]
        U = reward
        visited = set()
        
        # Retropropagación iterativa: desde el final de la trayectoria hasta el origen[cite: 9]
        for t in range(len(trajectory) - 1, -1, -1):
            s_hash, action, player = trajectory[t]
            state_action = (s_hash, action)
            
            # Condición de estricta de Primera Visita para mantener el modelo insesgado[cite: 9]
            if state_action not in visited:
                visited.add(state_action)
                
                if state_action not in self.n_table:
                    self.n_table[state_action] = 0
                    self.q_table[state_action] = 0.0
                    
                self.n_table[state_action] += 1
                
                # Actualización de media empírica de forma incremental constante[cite: 8, 9]
                error = U - self.q_table[state_action]
                self.q_table[state_action] += error / self.n_table[state_action]
            
            # Alternating Markov Games: Actualización Bipolar[cite: 10]
            # La utilidad para el oponente en el turno previo es estrictamente inversa[cite: 10].
            U = -self.gamma * U