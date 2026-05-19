import numpy as np
import time
from collections import defaultdict
from connect4.policy import Policy

class Connect4ADPAgent(Policy):
    """
    Agente de Connect 4 basado estrictamente en Adaptive Dynamic Programming (ADP).
    Utiliza el tiempo límite por turno para refinar la evaluación de la política (PE).
    """
    def __init__(self, max_time=4.0, gamma=0.9):
        super().__init__()
        self.max_time = max_time
        self.gamma = gamma
        
        # Estructuras de Datos de ADP (Diapositiva 17)
        self.S = set()                                                      # Conjunto S
        self.N = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # Conteo N[s][a][s']
        self.rewards_model = defaultdict(float)                             # Recompensas r
        self.P_hat = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # P_hat(s'|s,a)
        
        self.v_hat = defaultdict(float)                                     # v_hat
        self.q_hat = defaultdict(lambda: defaultdict(float))                # q_hat
        self.pi = defaultdict(lambda: defaultdict(float))                   # Política interna pi
        
        # Variables de rastreo de la trayectoria (Trial) en tiempo real
        self.last_state = None
        self.last_action = None

    def mount(self) -> None:
        """
        Línea 1 del algoritmo ADP: Inicialización de recursos memorizados.
        """
        self.last_state = None
        self.last_action = None
        # Nota: Retenemos S, N y P_hat a lo largo de las partidas porque el MDP del 
        # Connect 4 no cambia; así el conocimiento se transfiere (Diapositiva 18).

    def act(self, s: np.ndarray) -> int:
        start_time = time.time()
        
        # 1. Identificar estado actual de forma hashable
        current_state_key = s.tobytes()
        self.S.add(current_state_key)
        
        # Calcular columnas disponibles en el tablero actual
        available_cols = [c for c in range(7) if s[0, c] == 0]
        if not available_cols:
            return 0
            
        # 2. Registrar Transición del Mundo Real (Líneas 10 y 11 del algoritmo ADP)
        if self.last_state is not None and self.last_action is not None:
            # Capturamos la recompensa inmediata si el estado es terminal (heurística básica de fin de juego)
            reward = self._determine_immediate_reward(s)
            self.rewards_model[current_state_key] = reward
            
            # N[s_t-1][a_t-1][s_t] ++
            self.N[self.last_state][self.last_action][current_state_key] += 1
            
            # Re-calcular P_hat para este par estado-acción
            total_transitions = sum(self.N[self.last_state][self.last_action].values())
            for s_prime in self.N[self.last_state][self.last_action]:
                self.P_hat[self.last_state][self.last_action][s_prime] = (
                    self.N[self.last_state][self.last_action][s_prime] / total_transitions
                )

        # 3. Optimización por tiempo limitado: MDP-Policy-Evaluation (Línea 13)
        # Aprovechamos los 4 segundos para resolver Bellman sobre lo que conocemos del modelo
        self._time_bounded_policy_evaluation(start_time)
        
        # 4. Actualizar q_hat a partir de v_hat y P_hat (Línea 15)
        self._update_q_values_for_state(current_state_key)
        
        # 5. Selección de la acción bajo la política explotativa de q_hat
        # Si no conocemos q_hat del estado, recurrimos a una acción aleatoria (Exploración implícita)
        if current_state_key in self.q_hat and self.q_hat[current_state_key]:
            chosen_action = max(available_cols, key=lambda a: self.q_hat[current_state_key].get(a, 0.0))
        else:
            rng = np.random.default_rng()
            chosen_action = int(rng.choice(available_cols))
            
        # 6. Actualizar política interna pi para futuras evaluaciones de Bellman
        self.pi[current_state_key] = {a: (1.0 if a == chosen_action else 0.0) for a in available_cols}
        
        # Guardar historial para la transición del siguiente turno
        self.last_state = current_state_key
        self.last_action = chosen_action
        
        return chosen_action

    def _time_bounded_policy_evaluation(self, start_time, theta=1e-4):
        """
        Evaluación de la política adaptada al tiempo límite.
        v_hat(s) = r(s) + gamma * sum( P_hat(s'|s, pi(s)) * v_hat(s') )
        """
        while True:
            # Control estricto del tiempo límite por turno
            if time.time() - start_time > (self.max_time - 0.2): # Margen de seguridad de 0.2s
                break
                
            delta = 0
            for s in list(self.S):
                v_old = self.v_hat[s]
                expected_future_value = 0
                
                # sum_a pi(a|s) * sum_s' P(s'|s,a) * v(s')
                for a, prob_a in self.pi[s].items():
                    for s_prime, prob_transition in self.P_hat[s][a].items():
                        expected_future_value += prob_a * prob_transition * self.v_hat[s_prime]
                        
                self.v_hat[s] = self.rewards_model[s] + self.gamma * expected_future_value
                delta = max(delta, abs(v_old - self.v_hat[s]))
                
            if delta < theta:
                break

    def _update_q_values_for_state(self, s):
        """
        Deriva q_hat(s, a) basándose en la última actualización de v_hat (Línea 15).
        """
        for a in self.P_hat[s]:
            q_val = 0
            for s_prime, prob_transition in self.P_hat[s][a].items():
                q_val += prob_transition * self.v_hat[s_prime]
            self.q_hat[s][a] = self.rewards_model[s] + self.gamma * q_val

    def _determine_immediate_reward(self, board: np.ndarray) -> float:
        """
        Asigna utilidad básica al estado. En ADP, aprender las recompensas 'r' 
        es parte del proceso (Diapositiva 16).
        """
        # Si hay una condición de fin de juego detectable en el tablero recibido,
        # se puede retornar 1.0 (victoria) o -1.0 (derrota). De lo contrario, 0.0.
        return 0.0