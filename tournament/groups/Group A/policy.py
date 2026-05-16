import numpy as np
import math
import time
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override

class Aha(Policy):
    @override
    def mount(self) -> None:
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        rng = np.random.default_rng()
        available_cols = [c for c in range(7) if s[0, c] == 0]
        return int(rng.choice(available_cols))


class AlphaBetaAgent(Policy):
    """
    Agente que utiliza Minimax con Poda Alfa-Beta, 
    Profundidad Iterativa y Tablas de Transposición.
    """
    def __init__(self, max_time=4.0):
        super().__init__()
        self.max_time = max_time
        self.start_time = 0
        self.transposition_table = {}
        self.my_piece = 0

    @override
    def mount(self) -> None:
        self.transposition_table.clear()

    @override
    def act(self, s: np.ndarray) -> int:
        self.start_time = time.time()
        
        red_pieces = np.sum(s == -1)
        yellow_pieces = np.sum(s == 1)
        current_player = -1 if red_pieces == yellow_pieces else 1
        self.my_piece = current_player
        
        state = ConnectState(board=s, player=current_player)
        valid_locations = state.get_free_cols()
        
        # Fallback action
        best_col = valid_locations[len(valid_locations) // 2] if valid_locations else 0
        
        try:
            # Iterative deepening
            for depth in range(1, 42): 
                if time.time() - self.start_time > self.max_time:
                    break
                    
                col, score = self.minimax(state, depth, -math.inf, math.inf, True)
                
                if col is not None:
                    best_col = col
                    
                # Si encontramos victoria forzada o derrota forzada, no necesitamos buscar más profundo
                if score >= 900000 or score <= -900000:
                    break
                    
        except TimeoutError:
            # Se acabó el tiempo a mitad de una profundidad, usamos la mejor jugada de la profundidad anterior
            pass 
            
        return int(best_col)

    def minimax(self, state: ConnectState, depth: int, alpha: float, beta: float, maximizingPlayer: bool):
        if time.time() - self.start_time > self.max_time:
            raise TimeoutError()
            
        board_hash = hash((state.board.tobytes(), state.player))
        
        # TT Lookup
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
                if winner == self.my_piece:
                    return (None, 1000000 + depth) # Preferimos victorias más rápidas
                elif winner != 0:
                    return (None, -1000000 - depth) # Preferimos derrotas más lentas
                else: # Empate
                    return (None, 0)
            else:
                return (None, self.score_position(state.board, self.my_piece))
                
        valid_locations = state.get_free_cols()
        # Move ordering: evaluar el centro primero optimiza la poda Alfa-Beta
        valid_locations.sort(key=lambda x: abs(x - 3))
        
        orig_alpha = alpha
        best_col = valid_locations[0]

        if maximizingPlayer:
            value = -math.inf
            for col in valid_locations:
                next_state = state.transition(col)
                new_score = self.minimax(next_state, depth-1, alpha, beta, False)[1]
                
                if new_score > value:
                    value = new_score
                    best_col = col
                
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else: # Minimizing player
            value = math.inf
            for col in valid_locations:
                next_state = state.transition(col)
                new_score = self.minimax(next_state, depth-1, alpha, beta, True)[1]
                
                if new_score < value:
                    value = new_score
                    best_col = col
                    
                beta = min(beta, value)
                if alpha >= beta:
                    break
                    
        # TT Store
        flag = "EXACT"
        if value <= orig_alpha:
            flag = "UPPERBOUND"
        elif value >= beta:
            flag = "LOWERBOUND"
            
        self.transposition_table[board_hash] = (depth, value, flag, best_col)
        
        return best_col, value

    def score_position(self, board, piece):
        score = 0
        opp_piece = -piece
        
        # Center column preference
        center_array = board[:, 3]
        center_count = np.sum(center_array == piece)
        score += center_count * 30
        
        # Horizontal
        for r in range(6):
            row_array = board[r, :]
            for c in range(4):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, piece, opp_piece)
                
        # Vertical
        for c in range(7):
            col_array = board[:, c]
            for r in range(3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, piece, opp_piece)
                
        # Positive Diagonal
        for r in range(3):
            for c in range(4):
                window = [board[r+i, c+i] for i in range(4)]
                score += self.evaluate_window(window, piece, opp_piece)
                
        # Negative Diagonal
        for r in range(3):
            for c in range(4):
                window = [board[r+3-i, c+i] for i in range(4)]
                score += self.evaluate_window(window, piece, opp_piece)
                
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
            
        if piece_count == 3 and empty_count == 1:
            score += 50
        elif piece_count == 2 and empty_count == 2:
            score += 10
            
        if opp_count == 3 and empty_count == 1:
            score -= 80
            
        return score
