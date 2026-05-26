import numpy as np
import math
import time
from connect4.policy import Policy
from connect4.connect_state import ConnectState

def get_drop_row(board, col):
    for r in range(5, -1, -1):
        if board[r, col] == 0:
            return r
    return -1

def fast_win_check(board, player, row, col):
    # 1. Vertical
    if row <= 2:
        if board[row+1, col] == player and board[row+2, col] == player and board[row+3, col] == player:
            return True
    # 2. Horizontal
    count = 1
    for c in range(col-1, max(-1, col-4), -1):
        if board[row, c] == player: count += 1
        else: break
    for c in range(col+1, min(7, col+4)):
        if board[row, c] == player: count += 1
        else: break
    if count >= 4: return True
    # 3. Diagonal \
    count = 1
    for i in range(1, 4):
        if row-i >= 0 and col-i >= 0 and board[row-i, col-i] == player: count += 1
        else: break
    for i in range(1, 4):
        if row+i < 6 and col+i < 7 and board[row+i, col+i] == player: count += 1
        else: break
    if count >= 4: return True
    # 4. Diagonal /
    count = 1
    for i in range(1, 4):
        if row-i >= 0 and col+i < 7 and board[row-i, col+i] == player: count += 1
        else: break
    for i in range(1, 4):
        if row+i < 6 and col-i >= 0 and board[row+i, col-i] == player: count += 1
        else: break
    if count >= 4: return True
    return False

class MCTSTranspositionNode:
    def __init__(self, state: ConnectState, parent=None, action=None, depth=20, heuristics=True):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.available_actions = state.get_free_cols()
        self.depth = depth
        self.heuristics = heuristics
        self.center_col = 3
        
    def is_fully_expanded(self):
        return len(self.children) == len(self.available_actions)
    
    def is_terminal(self):
        return self.state.is_final()
    
    def best_child(self, exploration_weight):
        best_score = -float('inf')
        best_child = None
        for action, child in self.children.items():
            if child.visits == 0:
                return child
            win_rate = child.wins / child.visits
            exploration_term = math.sqrt(math.log(self.visits) / child.visits)
            heuristic_bonus = 0.0
            if self.heuristics:
                heuristic_bonus = (3 - abs(action - self.center_col)) * 0.01 
            score = win_rate + exploration_weight * exploration_term + heuristic_bonus
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def expand(self, transposition_table):
        untried_actions = [a for a in self.available_actions if a not in self.children]
        if not untried_actions: return None
        untried_actions.sort(key=lambda x: abs(x - self.center_col))
        action = untried_actions[0]
        
        next_state = self.state.transition(action)
        state_hash = self._hash_state(next_state)
        
        if state_hash in transposition_table:
            cached_stats = transposition_table[state_hash]
            child_node = MCTSTranspositionNode(next_state, parent=self, action=action, 
                                               depth=self.depth, heuristics=self.heuristics)
            child_node.visits = cached_stats['visits']
            child_node.wins = cached_stats['wins']
        else:
            child_node = MCTSTranspositionNode(next_state, parent=self, action=action, 
                                               depth=self.depth, heuristics=self.heuristics)
            
        self.children[action] = child_node
        return child_node

    def _hash_state(self, state):
        return hash((state.board.tobytes(), state.player))

    def _simulate(self, my_player):
        current_board = self.state.board.copy()
        current_player = self.state.player
        moves = 0
        winner = 0
        
        while moves < self.depth:
            available = [c for c in range(7) if current_board[0, c] == 0]
            if not available: break
                
            if self.heuristics:
                action = None
                for a in available:
                    r = get_drop_row(current_board, a)
                    current_board[r, a] = current_player
                    if fast_win_check(current_board, current_player, r, a):
                        action = a
                        current_board[r, a] = 0
                        break
                    current_board[r, a] = 0
                if action is None:
                    opp = -current_player
                    for a in available:
                        r = get_drop_row(current_board, a)
                        current_board[r, a] = opp
                        if fast_win_check(current_board, opp, r, a):
                            action = a
                            current_board[r, a] = 0
                            break
                        current_board[r, a] = 0
                if action is None:
                    safe_actions = []
                    for a in available:
                        r = get_drop_row(current_board, a)
                        if r > 0:
                            current_board[r, a] = current_player
                            current_board[r-1, a] = opp
                            win = fast_win_check(current_board, opp, r-1, a)
                            current_board[r-1, a] = 0
                            current_board[r, a] = 0
                            if win: continue
                        safe_actions.append(a)
                    cands = safe_actions if safe_actions else available
                    if self.center_col in cands: action = self.center_col
                    else: action = min(cands, key=lambda x: abs(x - self.center_col))
            else:
                if self.center_col in available: action = self.center_col
                else:
                    central = [c for c in available if 2 <= c <= 4]
                    if central: action = np.random.choice(central)
                    else: action = np.random.choice(available)
            
            r = get_drop_row(current_board, action)
            current_board[r, action] = current_player
            if fast_win_check(current_board, current_player, r, action):
                winner = current_player
                break
                
            current_player = -current_player
            moves += 1
            
        decay = 0.99 ** moves
        if winner == my_player: return 1.0 * decay
        elif winner == 0: return 0.5
        else: return 0.0

class AgenteOptimo(Policy):
    def __init__(self, num_simulations=50, exploration_weight=1.414, 
                 rollout_depth=25, heuristics_enabled=False, max_time=4.5):
        super().__init__()
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.heuristics_enabled = heuristics_enabled
        self.max_time = max_time
        self.transposition_table = {}

    def mount(self, timeout=None):
        self.transposition_table = {}
        if timeout is not None:
            self.max_time = float(timeout) * 0.85

    def act(self, s):
        start_time = time.time()
        red_pieces = np.sum(s == -1)
        yellow_pieces = np.sum(s == 1)
        current_player = -1 if red_pieces == yellow_pieces else 1
        initial_state = ConnectState(board=s, player=current_player)
        
        if initial_state.is_final():
            free = initial_state.get_free_cols()
            return int(free[0]) if free else 0

        # Transición al sistema Alpha-Beta Pruning en el endgame
        empty_squares = np.sum(s == 0)
        if empty_squares <= 14:
            score, best_action = self._minimax_alpha_beta(s.copy(), current_player, current_player, empty_squares, -float('inf'), float('inf'))
            if best_action is not None:
                return int(best_action)

        quick_move = self._check_immediate_moves(initial_state)
        if quick_move is not None:
            return int(quick_move)
            
        root = MCTSTranspositionNode(initial_state, depth=self.rollout_depth, heuristics=self.heuristics_enabled)
        
        simulations_run = 0
        while simulations_run < self.num_simulations:
            if time.time() - start_time > self.max_time:
                break
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand(self.transposition_table)
            reward = node._simulate(current_player)
            
            curr = node
            while curr is not None:
                curr.visits += 1
                if curr.parent is None:
                    curr.wins += reward
                elif curr.parent.state.player == current_player:
                    curr.wins += reward
                else:
                    curr.wins += (1.0 - reward)
                
                state_hash = curr._hash_state(curr.state)
                if state_hash not in self.transposition_table:
                    self.transposition_table[state_hash] = {'visits': 0, 'wins': 0.0}
                self.transposition_table[state_hash]['visits'] += 1
                self.transposition_table[state_hash]['wins'] += (reward if curr.parent and curr.parent.state.player == current_player else (1.0-reward))
                curr = curr.parent
            simulations_run += 1

        best_action = None
        max_visits = -1
        for action, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = action
                
        if best_action is None:
            if root.available_actions:
                best_action = root.available_actions[0]
            else:
                best_action = initial_state.get_free_cols()[0]
            
        return int(best_action)

    def _check_immediate_moves(self, state):
        available = state.get_free_cols()
        if not available: return None
        
        for action in available:
            r = get_drop_row(state.board, action)
            if r >= 0:
                state.board[r, action] = state.player
                win = fast_win_check(state.board, state.player, r, action)
                state.board[r, action] = 0
                if win: return action
                
        opponent = -state.player
        for action in available:
            r = get_drop_row(state.board, action)
            if r >= 0:
                state.board[r, action] = opponent
                win = fast_win_check(state.board, opponent, r, action)
                state.board[r, action] = 0
                if win: return action
                
        safe_actions = []
        for action in available:
            r = get_drop_row(state.board, action)
            if r > 0:
                state.board[r, action] = state.player
                state.board[r-1, action] = opponent
                win = fast_win_check(state.board, opponent, r-1, action)
                state.board[r-1, action] = 0
                state.board[r, action] = 0
                if win: continue
            safe_actions.append(action)
            
        if len(safe_actions) == 1:
            return safe_actions[0]
        if len(safe_actions) == 0:
            return available[0]
                
        return None

    def _minimax_alpha_beta(self, board, current_player, my_player, depth, alpha, beta):
        available = [c for c in range(7) if board[0, c] == 0]
        
        if not available or depth == 0:
            return 0, None  # Empate o profundidad máxima (que en este caso es 0)
            
        # Ordenar movimientos (centro primero) para mayor probabilidad de poda (pruning)
        available.sort(key=lambda x: abs(x - 3))
        is_maximizing = (current_player == my_player)
        best_action = available[0]
        
        if is_maximizing:
            max_eval = -float('inf')
            for action in available:
                r = get_drop_row(board, action)
                board[r, action] = current_player
                
                if fast_win_check(board, current_player, r, action):
                    board[r, action] = 0
                    return 1000 + depth, action
                
                eval_score, _ = self._minimax_alpha_beta(board, -current_player, my_player, depth-1, alpha, beta)
                board[r, action] = 0
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                    
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta pruning
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in available:
                r = get_drop_row(board, action)
                board[r, action] = current_player
                
                if fast_win_check(board, current_player, r, action):
                    board[r, action] = 0
                    return -1000 - depth, action
                
                eval_score, _ = self._minimax_alpha_beta(board, -current_player, my_player, depth-1, alpha, beta)
                board[r, action] = 0
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                    
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta pruning
            return min_eval, best_action
