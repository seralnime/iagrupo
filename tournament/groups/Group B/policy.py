import numpy as np
import math
import time
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override

class Hello(Policy):
    @override
    def mount(self) -> None:
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        rng = np.random.default_rng()
        available_cols = [c for c in range(7) if s[0, c] == 0]
        return int(rng.choice(available_cols))


class NegaScoutAgent(Policy):
    """
    Agente avanzado que utiliza NegaScout (Principal Variation Search),
    Heurística de Historia, y Tablas de Transposición.
    """
    def __init__(self, max_time=4.0):
        super().__init__()
        self.max_time = max_time
        self.start_time = 0
        self.transposition_table = {}
        # Historial para ordenamiento de movimientos: history_table[jugador][columna]
        self.history_table = {1: [0]*7, -1: [0]*7}

    @override
    def mount(self) -> None:
        self.transposition_table.clear()
        self.history_table = {1: [0]*7, -1: [0]*7}

    @override
    def act(self, s: np.ndarray) -> int:
        self.start_time = time.time()
        
        red_pieces = np.sum(s == -1)
        yellow_pieces = np.sum(s == 1)
        current_player = -1 if red_pieces == yellow_pieces else 1
        
        state = ConnectState(board=s, player=current_player)
        valid_locations = state.get_free_cols()
        
        best_col = valid_locations[len(valid_locations) // 2] if valid_locations else 0
        
        try:
            # Profundidad Iterativa
            for depth in range(1, 42): 
                if time.time() - self.start_time > self.max_time:
                    break
                    
                col, score = self.negascout(state, depth, -math.inf, math.inf)
                
                if col is not None:
                    best_col = col
                    
                # Cortar la búsqueda si encontramos una victoria o derrota forzada
                if score >= 900000 or score <= -900000:
                    break
                    
        except TimeoutError:
            # Si se acaba el tiempo, nos quedamos con la mejor jugada de la profundidad anterior
            pass 
            
        return int(best_col)

    def negascout(self, state: ConnectState, depth: int, alpha: float, beta: float):
        if time.time() - self.start_time > self.max_time:
            raise TimeoutError()
            
        board_hash = hash((state.board.tobytes(), state.player))
        
        # Transposition Table Lookup
        if board_hash in self.transposition_table:
            cached_depth, cached_score, cached_flag, cached_col = self.transposition_table[board_hash]
            if cached_depth >= depth:
                if cached_flag == "EXACT":
                    return cached_col, cached_score
                elif cached_flag == "LOWERBOUND":
                    alpha = max(alpha, cached_score)
                elif cached_flag == "UPPERBOUND":
                    beta = min(beta, cached_score)
                
                if alpha >= beta:
                    return cached_col, cached_score

        winner = state.get_winner()
        is_terminal = winner != 0 or not any(state.board[0] == 0)
        
        if depth == 0 or is_terminal:
            if is_terminal:
                if winner == 0: # Empate
                    return (None, 0)
                else:
                    # El jugador anterior ganó. Por lo tanto, el estado actual es una derrota forzada.
                    # Restamos la profundidad para preferir perder más tarde (o ganar más rápido).
                    return (None, -1000000 - depth)
            else:
                # Evaluar heurísticamente desde la perspectiva del jugador que tiene el turno
                return (None, self.score_position(state.board, state.player))
                
        valid_locations = state.get_free_cols()
        player = state.player
        
        # Move ordering: Heurística de Historia + Preferencia Central
        def order_score(col):
            center_bonus = (3 - abs(col - 3)) * 10
            return center_bonus + self.history_table[player][col]
            
        valid_locations.sort(key=order_score, reverse=True)
        
        orig_alpha = alpha
        best_col = valid_locations[0]
        value = -math.inf
        
        for i, col in enumerate(valid_locations):
            next_state = state.transition(col)
            
            if i == 0:
                # Búsqueda con ventana completa para el primer movimiento (Principal Variation)
                _, child_score = self.negascout(next_state, depth-1, -beta, -alpha)
                current_score = -child_score
            else:
                # Búsqueda con ventana nula (PVS) para demostrar que los demás movimientos son peores
                _, child_score = self.negascout(next_state, depth-1, -alpha - 1, -alpha)
                current_score = -child_score
                
                # Si el movimiento resulta ser mejor de lo esperado, volver a buscar con ventana completa
                if alpha < current_score < beta:
                    _, child_score = self.negascout(next_state, depth-1, -beta, -current_score)
                    current_score = -child_score
                    
            if current_score > value:
                value = current_score
                best_col = col
                
            alpha = max(alpha, value)
            if alpha >= beta:
                # History Heuristic: Registrar el movimiento que causó el corte
                self.history_table[player][col] += depth * depth
                break
                
        # Transposition Table Store
        flag = "EXACT"
        if value <= orig_alpha:
            flag = "UPPERBOUND"
        elif value >= beta:
            flag = "LOWERBOUND"
            
        self.transposition_table[board_hash] = (depth, value, flag, best_col)
        
        return best_col, value

    def score_position(self, board, player):
        score = 0
        opp_player = -player
        
        # Preferencia central estática
        center_array = board[:, 3]
        score += np.sum(center_array == player) * 30
        
        # Evaluaciones de ventanas horizontales
        for r in range(6):
            row_array = board[r, :]
            for c in range(4):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, player, opp_player)
                
        # Verticales
        for c in range(7):
            col_array = board[:, c]
            for r in range(3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, player, opp_player)
                
        # Diagonales Positivas
        for r in range(3):
            for c in range(4):
                window = [board[r+i, c+i] for i in range(4)]
                score += self.evaluate_window(window, player, opp_player)
                
        # Diagonales Negativas
        for r in range(3):
            for c in range(4):
                window = [board[r+3-i, c+i] for i in range(4)]
                score += self.evaluate_window(window, player, opp_player)
                
        return score

    def evaluate_window(self, window, piece, opp_piece):
        score = 0
        if isinstance(window, list):
            piece_count = window.count(piece)
            empty_count = window.count(0)
            opp_count = window.count(opp_piece)
        else:
            piece_count = np.sum(window == piece)
            empty_count = np.sum(window == 0)
            opp_count = np.sum(window == opp_piece)
            
        # Puntos por amenaza propia
        if piece_count == 3 and empty_count == 1:
            score += 50
        elif piece_count == 2 and empty_count == 2:
            score += 10
            
        # Penalización por amenaza del oponente
        if opp_count == 3 and empty_count == 1:
            score -= 80
            
        return score
