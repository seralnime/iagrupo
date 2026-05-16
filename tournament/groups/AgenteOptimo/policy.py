import numpy as np
import math
import time
from connect4.policy import Policy
from connect4.connect_state import ConnectState

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
        
        # Center column heuristic bias
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
            
            # UCB1 Formula
            win_rate = child.wins / child.visits
            exploration_term = math.sqrt(math.log(self.visits) / child.visits)
            
            # Heuristic bonus: prefer center columns slightly when exploring
            heuristic_bonus = 0.0
            if self.heuristics:
                dist_to_center = abs(action - self.center_col)
                # Max dist is 3, bonus ranges from 0.0 to 0.03
                heuristic_bonus = (3 - dist_to_center) * 0.01 
            
            score = win_rate + exploration_weight * exploration_term + heuristic_bonus
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def expand(self, transposition_table):
        untried_actions = [a for a in self.available_actions if a not in self.children]
        if not untried_actions:
            return None
            
        # Try to expand center columns first
        untried_actions.sort(key=lambda x: abs(x - self.center_col))
        action = untried_actions[0]
        
        next_state = self.state.transition(action)
        state_hash = self._hash_state(next_state)
        
        # Use transposition table to avoid re-expanding known states
        if state_hash in transposition_table:
            # We found a similar state, but we must create a new node in this tree path
            # However, we can initialize its statistics with the known ones
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
        current_state = ConnectState(board=self.state.board.copy(), player=self.state.player)
        moves = 0
        
        while moves < self.depth:
            if current_state.is_final():
                break
                
            available = current_state.get_free_cols()
            if not available:
                break
                
            if self.heuristics:
                action = self._select_heuristic_action(current_state, available)
            else:
                action = self._select_fast_action(current_state, available)
                
            try:
                current_state = current_state.transition(action)
                moves += 1
            except ValueError:
                break
                
        winner = current_state.get_winner()
        
        # Decay factor: prefer faster wins and slower losses
        decay = 0.99 ** moves
        
        if winner == my_player:
            return 1.0 * decay
        elif winner == 0:
            return 0.5
        else:
            return 0.0

    def _select_heuristic_action(self, state, available_actions):
        # 1. Immediate Win
        for action in available_actions:
            if state.transition(action).get_winner() == state.player:
                return action
                
        # 2. Immediate Block
        opponent = -state.player
        test_state = ConnectState(board=state.board.copy(), player=opponent)
        for action in available_actions:
            if test_state.transition(action).get_winner() == opponent:
                return action
                
        # 3. Filter safe actions (avoid playing below an opponent's winning spot)
        safe_actions = []
        for action in available_actions:
            next_state = state.transition(action)
            if action in next_state.get_free_cols():
                test_opp_state = ConnectState(board=next_state.board.copy(), player=opponent)
                if test_opp_state.transition(action).get_winner() == opponent:
                    continue # Suicide move
            safe_actions.append(action)
            
        # If all moves are suicide, just pick from available
        candidate_actions = safe_actions if safe_actions else available_actions
                
        # 4. Center preference among candidates
        if self.center_col in candidate_actions:
            return self.center_col
            
        return min(candidate_actions, key=lambda x: abs(x - self.center_col))

    def _select_fast_action(self, state, available_actions):
        if self.center_col in available_actions:
            return self.center_col
        central_actions = [col for col in available_actions if 2 <= col <= 4]
        if central_actions:
            return np.random.choice(central_actions)
        return np.random.choice(available_actions)


class AgenteOptimo(Policy):
    """
    Agente MCTS Mejorado con Transposition Tables y Heurísticas.
    Diseñado para el torneo de Connect-4.
    """
    def __init__(self, num_simulations=500, exploration_weight=1.414, 
                 rollout_depth=25, heuristics_enabled=True, max_time=4.5):
        super().__init__()
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.heuristics_enabled = heuristics_enabled
        self.max_time = max_time # Time limit in seconds
        self.transposition_table = {}

    def mount(self):
        # Clear the transposition table between matches
        self.transposition_table = {}

    def act(self, s):
        start_time = time.time()
        
        red_pieces = np.sum(s == -1)
        yellow_pieces = np.sum(s == 1)
        current_player = -1 if red_pieces == yellow_pieces else 1
        
        initial_state = ConnectState(board=s, player=current_player)
        
        # 0. Check immediate moves to save time
        quick_move = self._check_immediate_moves(initial_state)
        if quick_move is not None:
            return int(quick_move)
            
        root = MCTSTranspositionNode(initial_state, depth=self.rollout_depth, heuristics=self.heuristics_enabled)
        
        simulations_run = 0
        while simulations_run < self.num_simulations:
            # Time constraint check
            if time.time() - start_time > self.max_time:
                break
                
            node = root
            
            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
                
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand(self.transposition_table)
                
            # Simulation
            reward = node._simulate(current_player)
            
            # Backpropagation
            curr = node
            while curr is not None:
                curr.visits += 1
                if curr.parent is None:
                    curr.wins += reward
                elif curr.parent.state.player == current_player:
                    curr.wins += reward
                else:
                    curr.wins += (1.0 - reward)
                    
                # Update Transposition Table
                state_hash = curr._hash_state(curr.state)
                if state_hash not in self.transposition_table:
                    self.transposition_table[state_hash] = {'visits': 0, 'wins': 0.0}
                self.transposition_table[state_hash]['visits'] += 1
                self.transposition_table[state_hash]['wins'] += (reward if curr.parent and curr.parent.state.player == current_player else (1.0-reward))
                
                curr = curr.parent
                
            simulations_run += 1

        # Select the best action based on visit counts
        best_action = None
        max_visits = -1
        for action, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = action
                
        if best_action is None:
            best_action = initial_state.get_free_cols()[0]
            
        return int(best_action)

    def _check_immediate_moves(self, state):
        available = state.get_free_cols()
        
        # Win immediately
        for action in available:
            if state.transition(action).get_winner() == state.player:
                return action
                
        # Block immediately
        opponent = -state.player
        test_state = ConnectState(board=state.board.copy(), player=opponent)
        for action in available:
            if test_state.transition(action).get_winner() == opponent:
                return action
        # Avoid MCTS root expanding suicide moves if possible
        safe_actions = []
        for action in available:
            next_state = state.transition(action)
            if action in next_state.get_free_cols():
                test_opp_state = ConnectState(board=next_state.board.copy(), player=opponent)
                if test_opp_state.transition(action).get_winner() == opponent:
                    continue
            safe_actions.append(action)
            
        # If we have safe actions, restrict MCTS to only these by hacking the initial state?
        # A simpler way is to just let MCTS figure it out, but returning None means MCTS will handle it.
        # But if there's only 1 safe action left, we MUST play it to avoid losing.
        if len(safe_actions) == 1:
            return safe_actions[0]
            
        # If no safe actions, play anything, we've probably lost
        if len(safe_actions) == 0:
            return available[0]
                
        return None
