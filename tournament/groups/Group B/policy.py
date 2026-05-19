import numpy as np
import time
from collections import defaultdict
from connect4.policy import Policy
from connect4.connect_state import ConnectState # Usamos tu clase de estado para el Reward Shaping
from typing import override

class Connect4ADPAgent(Policy):
    """
    Agente de Connect 4 basado estrictamente en Adaptive Dynamic Programming (ADP).
    Incluye Reward Shaping para garantizar convergencia rápida contra agentes aleatorios.
    """
    def __init__(self, max_time=4.0, gamma=0.9):
        super().__init__()
        self.max_time = max_time
        self.gamma = gamma
        
        # Estructuras de Datos de ADP (Diapositiva 17)
        self.S = set()
        self.N = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.rewards_model = defaultdict(float)
        self.P_hat = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        self.v_hat = defaultdict(float)
        self.q_hat = defaultdict(lambda: defaultdict(float))
        self.pi = defaultdict(lambda: defaultdict(float))
        
        # Variables de rastreo de la trayectoria
        self.last_state = None
        self.last_action = None
        self.my_player_id = None # Identificador para dar la recompensa al jugador correcto

    @override
    def mount(self, timeout: float = 4.0) -> None:
        """
        Línea 1 del algoritmo ADP.
        Aceptamos el parámetro `timeout` que el autograder inyecta para evitar el TypeError.
        """
        if timeout is not None:
            self.max_time = timeout
            
        self.last_state = None
        self.last_action = None
        self.my_player_id = None 

    @override
    def act(self, s: np.ndarray) -> int:
        start_time = time.time()
        
        # 1. Identificar nuestra pieza (solo ocurre en el primer turno de la partida)
        if self.my_player_id is None:
            red_pieces = np.sum(s == -1)
            yellow_pieces = np.sum(s == 1)
            self.my_player_id = -1 if red_pieces == yellow_pieces else 1
            
        current_state_key = s.tobytes()
        self.S.add(current_state_key)
        
        available_cols = [c for c in range(7) if s[0, c] == 0]
        if not available_cols:
            return 0
            
        # 2. Registrar Transición del Mundo Real y Reward Shaping
        if self.last_state is not None and self.last_action is not None:
            # Reward Shaping: Asignamos recompensas sintéticas (Diapositiva 26)
            reward = self._determine_immediate_reward(s)
            self.rewards_model[current_state_key] = reward
            
            # Conteo de transiciones y actualización de probabilidades estimadas
            self.N[self.last_state][self.last_action][current_state_key] += 1
            total_transitions = sum(self.N[self.last_state][self.last_action].values())
            for s_prime in self.N[self.last_state][self.last_action]:
                self.P_hat[self.last_state][self.last_action][s_prime] = (
                    self.N[self.last_state][self.last_action][s_prime] / total_transitions
                )

        # 3. Optimización por tiempo limitado: MDP-Policy-Evaluation
        self._time_bounded_policy_evaluation(start_time)
        
        # 4. Actualizar q_hat a partir de v_hat y P_hat
        self._update_q_values_for_state(current_state_key)
        
        # 5. Selección de la acción (Exploración implícita guiada al centro)
        if current_state_key in self.q_hat and self.q_hat[current_state_key]:
            chosen_action = max(available_cols, key=lambda a: self.q_hat[current_state_key].get(a, 0.0))
        else:
            # Para acelerar el aprendizaje frente a la aleatoriedad, probamos las columnas 
            # centrales primero cuando el estado es desconocido. Esto aumenta drásticamente
            # las probabilidades de conectar 4 rápido.
            center_preference = [3, 2, 4, 1, 5, 0, 6]
            chosen_action = next(c for c in center_preference if c in available_cols)
            
        # 6. Actualizar política interna pi para futuras evaluaciones de Bellman
        self.pi[current_state_key] = {a: (1.0 if a == chosen_action else 0.0) for a in available_cols}
        
        self.last_state = current_state_key
        self.last_action = chosen_action
        
        return chosen_action

    def _time_bounded_policy_evaluation(self, start_time, theta=1e-4):
        """
        Evaluación de la política adaptada al tiempo límite.
        """
        while True:
            if time.time() - start_time > (self.max_time - 0.2):
                break
                
            delta = 0
            for s in list(self.S):
                v_old = self.v_hat[s]
                expected_future_value = 0
                
                for a, prob_a in self.pi[s].items():
                    for s_prime, prob_transition in self.P_hat[s][a].items():
                        expected_future_value += prob_a * prob_transition * self.v_hat[s_prime]
                        
                self.v_hat[s] = self.rewards_model[s] + self.gamma * expected_future_value
                delta = max(delta, abs(v_old - self.v_hat[s]))
                
            if delta < theta:
                break

    def _update_q_values_for_state(self, s):
        """
        Deriva q_hat(s, a) basándose en la última actualización de v_hat.
        """
        for a in self.P_hat[s]:
            q_val = 0
            for s_prime, prob_transition in self.P_hat[s][a].items():
                q_val += prob_transition * self.v_hat[s_prime]
            self.q_hat[s][a] = self.rewards_model[s] + self.gamma * q_val

    def _determine_immediate_reward(self, board: np.ndarray) -> float:
        """
        Implementación estricta de Reward Shaping (Diapositiva 26).
        Se inyectan recompensas sintéticas en estados terminales conocidos
        para propagar el valor rápidamente por Bellman.
        """
        state = ConnectState(board=board, player=self.my_player_id)
        winner = state.get_winner()
        
        if winner == self.my_player_id:
            return 100.0  # Gran recompensa por ganar
        elif winner != 0 and winner != self.my_player_id:
            return -100.0 # Gran penalización por perder
            
        return 0.0