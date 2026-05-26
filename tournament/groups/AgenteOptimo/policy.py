import numpy as np
import math
import time
from connect4.policy import Policy
from connect4.connect_state import ConnectState

def get_drop_row(board, col):
    """
    Función auxiliar para encontrar la fila más baja disponible en una columna dada.
    Itera desde la parte inferior (fila 5) hacia arriba (fila 0).
    Retorna el índice de la fila o -1 si la columna está llena.
    """
    for r in range(5, -1, -1):
        if board[r, col] == 0:
            return r
    return -1

def fast_win_check(board, player, row, col):
    """
    Verifica rápidamente si el jugador 'player' acaba de ganar al colocar
    una ficha en la posición (row, col). Solo comprueba las 4 direcciones
    alrededor de esa ficha recién colocada, en lugar de escanear todo el tablero.
    Retorna True si hay victoria, False en caso contrario.
    """
    # 1. Comprobación Vertical (Hacia abajo desde la ficha)
    if row <= 2:
        if board[row+1, col] == player and board[row+2, col] == player and board[row+3, col] == player:
            return True
            
    # 2. Comprobación Horizontal (Izquierda a Derecha)
    count = 1
    # Mirar hacia la izquierda
    for c in range(col-1, max(-1, col-4), -1):
        if board[row, c] == player: count += 1
        else: break
    # Mirar hacia la derecha
    for c in range(col+1, min(7, col+4)):
        if board[row, c] == player: count += 1
        else: break
    if count >= 4: return True
    
    # 3. Comprobación Diagonal Descendente (\)
    count = 1
    # Mirar hacia arriba-izquierda
    for i in range(1, 4):
        if row-i >= 0 and col-i >= 0 and board[row-i, col-i] == player: count += 1
        else: break
    # Mirar hacia abajo-derecha
    for i in range(1, 4):
        if row+i < 6 and col+i < 7 and board[row+i, col+i] == player: count += 1
        else: break
    if count >= 4: return True
    
    # 4. Comprobación Diagonal Ascendente (/)
    count = 1
    # Mirar hacia arriba-derecha
    for i in range(1, 4):
        if row-i >= 0 and col+i < 7 and board[row-i, col+i] == player: count += 1
        else: break
    # Mirar hacia abajo-izquierda
    for i in range(1, 4):
        if row+i < 6 and col-i >= 0 and board[row+i, col-i] == player: count += 1
        else: break
    if count >= 4: return True
    
    # Si ninguna comprobación superó 4 fichas alineadas, no hay victoria
    return False

class MCTSTranspositionNode:
    """
    Representa un nodo en el árbol de búsqueda Monte Carlo (MCTS).
    Esta versión está preparada para trabajar en conjunto con Tablas de Transposición,
    evitando recalcular estados a los que se puede llegar por diferente orden de jugadas.
    """
    def __init__(self, state: ConnectState, parent=None, action=None, depth=20, heuristics=True):
        self.state = state               # Estado del tablero en este nodo
        self.parent = parent             # Nodo padre (el estado anterior)
        self.action = action             # La acción (columna) que llevó a este nodo
        self.children = {}               # Nodos hijos (claves: acción, valores: MCTSTranspositionNode)
        self.visits = 0                  # Número de veces que este nodo ha sido visitado en las simulaciones
        self.wins = 0                    # Sumatoria de recompensas/victorias obtenidas desde este nodo
        self.available_actions = state.get_free_cols() # Columnas donde aún se puede jugar
        self.depth = depth               # Profundidad máxima permitida para los rollouts
        self.heuristics = heuristics     # Bandera para habilitar/deshabilitar heurísticas en rollouts
        self.center_col = 3              # Columna central (la más valiosa en Connect-4)
        
    def is_fully_expanded(self):
        """
        Retorna True si todas las jugadas posibles desde este nodo ya han sido
        añadidas como hijos en el árbol (es decir, ya fueron probadas al menos una vez).
        """
        return len(self.children) == len(self.available_actions)
    
    def is_terminal(self):
        """
        Retorna True si este estado es el fin del juego (alguien ganó o empate).
        """
        return self.state.is_final()
    
    def best_child(self, exploration_weight):
        """
        Selecciona el mejor nodo hijo utilizando la fórmula UCB1 (Upper Confidence Bound).
        Balancea la 'explotación' (elegir el movimiento con mayor win_rate)
        con la 'exploración' (elegir movimientos menos visitados).
        """
        best_score = -float('inf')
        best_child = None
        for action, child in self.children.items():
            # Si un nodo hijo nunca ha sido visitado, tiene prioridad máxima
            if child.visits == 0:
                return child
                
            # Explotación: porcentaje de victorias
            win_rate = child.wins / child.visits
            # Exploración: bonificación matemática para nodos menos visitados
            exploration_term = math.sqrt(math.log(self.visits) / child.visits)
            
            # Heurística: dar un pequeño bonus a los movimientos más centrales
            heuristic_bonus = 0.0
            if self.heuristics:
                heuristic_bonus = (3 - abs(action - self.center_col)) * 0.01 
                
            # Puntaje total UCB1 modificado
            score = win_rate + exploration_weight * exploration_term + heuristic_bonus
            
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def expand(self, transposition_table):
        """
        Expande el nodo actual creando un nuevo hijo para una jugada que aún no se ha probado.
        """
        # Filtrar las acciones que aún no están en los hijos
        untried_actions = [a for a in self.available_actions if a not in self.children]
        if not untried_actions: return None
        
        # Heurística: Priorizar jugar en las columnas más centrales primero
        untried_actions.sort(key=lambda x: abs(x - self.center_col))
        action = untried_actions[0]
        
        # Generar el nuevo estado después de aplicar la acción
        next_state = self.state.transition(action)
        # Calcular el hash único del nuevo tablero
        state_hash = self._hash_state(next_state)
        
        # ====== TABLA DE TRANSPOSICIÓN ======
        # Si este tablero exacto ya se evaluó en otra rama del árbol,
        # recuperamos sus estadísticas en lugar de empezar de 0.
        if state_hash in transposition_table:
            cached_stats = transposition_table[state_hash]
            child_node = MCTSTranspositionNode(next_state, parent=self, action=action, 
                                               depth=self.depth, heuristics=self.heuristics)
            child_node.visits = cached_stats['visits']
            child_node.wins = cached_stats['wins']
        else:
            # Estado nuevo, inicialización estándar en 0
            child_node = MCTSTranspositionNode(next_state, parent=self, action=action, 
                                               depth=self.depth, heuristics=self.heuristics)
            
        # Registrar el nuevo hijo
        self.children[action] = child_node
        return child_node

    def _hash_state(self, state):
        """
        Genera un identificador único (hash) rápido para un tablero dado
        utilizando sus bytes brutos y de quién es el turno.
        """
        return hash((state.board.tobytes(), state.player))

    def _simulate(self, my_player):
        """
        Fase de Rollout: Juega una partida (simulada hacia el futuro) desde este estado
        hasta alcanzar una victoria, derrota o el límite de profundidad (depth).
        Retorna la recompensa de la simulación.
        """
        current_board = self.state.board.copy()
        current_player = self.state.player
        moves = 0
        winner = 0
        
        while moves < self.depth:
            # Detectar columnas disponibles
            available = [c for c in range(7) if current_board[0, c] == 0]
            if not available: break # Tablero lleno (empate)
                
            if self.heuristics:
                action = None
                
                # 1. Comprobar si podemos ganar Inmediatamente
                for a in available:
                    r = get_drop_row(current_board, a)
                    current_board[r, a] = current_player
                    if fast_win_check(current_board, current_player, r, a):
                        action = a
                        current_board[r, a] = 0
                        break
                    current_board[r, a] = 0
                    
                # 2. Comprobar si el oponente va a ganar y hay que bloquearlo
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
                        
                # 3. Evitar hacer jugadas que le den la victoria al oponente justo encima de nuestra ficha
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
                            if win: continue # Jugada suicida o peligrosa, la evitamos
                        safe_actions.append(a)
                        
                    # Filtrar posibles acciones, prefiriendo siempre el centro
                    cands = safe_actions if safe_actions else available
                    if self.center_col in cands: action = self.center_col
                    else: action = min(cands, key=lambda x: abs(x - self.center_col))
            else:
                # Simulación puramente aleatoria (sesgada ligeramente al centro)
                if self.center_col in available: action = self.center_col
                else:
                    central = [c for c in available if 2 <= c <= 4]
                    if central: action = np.random.choice(central)
                    else: action = np.random.choice(available)
            
            # Ejecutar la acción elegida en el tablero simulado temporal
            r = get_drop_row(current_board, action)
            current_board[r, action] = current_player
            
            # Verificar si la simulación acaba de terminar (alguien ganó)
            if fast_win_check(current_board, current_player, r, action):
                winner = current_player
                break
                
            # Cambiar de turno para el siguiente paso de la simulación
            current_player = -current_player
            moves += 1
            
        # Recompensa con decaimiento por tiempo: prefiere ganar rápido (en menos movimientos)
        decay = 0.99 ** moves
        if winner == my_player: return 1.0 * decay
        elif winner == 0: return 0.5  # Empate otorga media recompensa
        else: return 0.0              # Derrota otorga 0 recompensa

class AgenteOptimo(Policy):
    """
    La política principal del Agente. Extiende de la clase base Policy.
    Implementa el control de tiempo, MCTS y transición al sistema Alpha-Beta para endgames.
    """
    def __init__(self, num_simulations=50, exploration_weight=1.414, 
                 rollout_depth=25, heuristics_enabled=False, max_time=4.5):
        super().__init__()
        self.num_simulations = num_simulations       # Número máximo de simulaciones MCTS por turno
        self.exploration_weight = exploration_weight # Parámetro de exploración para el UCB1 (teóricamente raíz de 2)
        self.rollout_depth = rollout_depth           # Límite de profundidad para rollouts
        self.heuristics_enabled = heuristics_enabled # Activar heurísticas en rollouts
        self.max_time = max_time                     # Tiempo límite de pensamiento por turno
        self.transposition_table = {}                # Caché global del árbol para no repetir nodos

    def mount(self, timeout=None):
        """
        Se ejecuta al iniciar una nueva partida.
        Limpia la memoria del agente (transposition table) y configura el timeout basado en el entorno.
        """
        self.transposition_table = {}
        if timeout is not None:
            # Usar solo el 85% del tiempo permitido para evitar descalificación por timeout en torneos
            self.max_time = float(timeout) * 0.85

    def act(self, s):
        """
        Función núcleo que se invoca cada turno para que el agente decida su movimiento final.
        Recibe 's', el arreglo (tablero numpy) del estado actual.
        """
        start_time = time.time()
        
        # Determinar de quién es el turno contando las fichas del tablero
        red_pieces = np.sum(s == -1)
        yellow_pieces = np.sum(s == 1)
        current_player = -1 if red_pieces == yellow_pieces else 1
        initial_state = ConnectState(board=s, player=current_player)
        
        # Caso base: Si el juego ya está terminado de alguna forma, devolver cualquier cosa
        if initial_state.is_final():
            free = initial_state.get_free_cols()
            return int(free[0]) if free else 0

        # ======= ENDGAME: Transición al sistema Alpha-Beta Pruning ======
        # Si quedan pocas casillas vacías (14 o menos), el cálculo estadístico (MCTS) es impreciso.
        # Es mejor usar fuerza bruta matemática orientada (Minimax) para garantizar victorias exactas.
        empty_squares = np.sum(s == 0)
        if empty_squares <= 14:
            score, best_action = self._minimax_alpha_beta(s.copy(), current_player, current_player, empty_squares, -float('inf'), float('inf'))
            if best_action is not None:
                return int(best_action)

        # Filtro inicial: Buscar jugadas ganadoras urgentes o de bloqueo forzado antes de iniciar MCTS.
        # Ahorra valiosos milisegundos que no se desperdiciarán simulando jugadas obvias.
        quick_move = self._check_immediate_moves(initial_state)
        if quick_move is not None:
            return int(quick_move)
            
        # ======= BÚSQUEDA MCTS ======
        # Crear la raíz del árbol con el estado actual
        root = MCTSTranspositionNode(initial_state, depth=self.rollout_depth, heuristics=self.heuristics_enabled)
        
        simulations_run = 0
        while simulations_run < self.num_simulations:
            # Control de tiempo: Salir si nos quedamos sin tiempo
            if time.time() - start_time > self.max_time:
                break
                
            node = root
            
            # 1. SELECCIÓN: Bajar por el árbol usando UCB1 hasta hallar un nodo con hijos sin explorar
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
                
            # 2. EXPANSIÓN: Si el nodo no está expandido por completo, añadir un nuevo hijo para un movimiento inexplorado
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand(self.transposition_table)
                
            # 3. SIMULACIÓN (Rollout): Jugar una partida simulada rápida desde el nodo seleccionado
            reward = node._simulate(current_player)
            
            # 4. RETROPROPAGACIÓN (Backpropagation): Subir la recompensa desde este nodo hasta la raíz
            curr = node
            while curr is not None:
                curr.visits += 1
                # Si el turno en este nodo nos pertenecía a nosotros (el agente), sumar la victoria
                if curr.parent is None:
                    curr.wins += reward
                elif curr.parent.state.player == current_player:
                    curr.wins += reward
                # Si el turno era del oponente, sumar la recompensa invertida (1 - reward)
                else:
                    curr.wins += (1.0 - reward)
                
                # Actualizar también la tabla de transposición global para que otras ramas aprovechen el cálculo
                state_hash = curr._hash_state(curr.state)
                if state_hash not in self.transposition_table:
                    self.transposition_table[state_hash] = {'visits': 0, 'wins': 0.0}
                self.transposition_table[state_hash]['visits'] += 1
                self.transposition_table[state_hash]['wins'] += (reward if curr.parent and curr.parent.state.player == current_player else (1.0-reward))
                
                # Subir de nivel (ir al padre)
                curr = curr.parent
            simulations_run += 1

        # ====== DECISIÓN FINAL ======
        # Escoger el hijo más visitado (que es estadísticamente el más robusto según MCTS)
        best_action = None
        max_visits = -1
        for action, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = action
                
        # Por seguridad (fallback), por si no se pudo decidir ningún nodo o no hubieron simulaciones
        if best_action is None:
            if root.available_actions:
                best_action = root.available_actions[0]
            else:
                best_action = initial_state.get_free_cols()[0]
            
        return int(best_action)

    def _check_immediate_moves(self, state):
        """
        Heurística determinista principal que se ejecuta antes del MCTS.
        1. Comprueba si tenemos un movimiento para ganar instantáneamente.
        2. Comprueba si el oponente va a ganar y hay que bloquearlo forzosamente.
        3. Identifica movimientos "seguros" (aquellos que no le dan la victoria al rival arriba nuestro).
        """
        available = state.get_free_cols()
        if not available: return None
        
        # 1. Ganar instantáneamente: si una ficha gana el juego, tirarla sin pensar
        for action in available:
            r = get_drop_row(state.board, action)
            if r >= 0:
                state.board[r, action] = state.player
                win = fast_win_check(state.board, state.player, r, action)
                state.board[r, action] = 0
                if win: return action
                
        # 2. Bloquear al oponente: si el oponente gana tirando en 'x' columna, tenemos que tirar ahí forzosamente
        opponent = -state.player
        for action in available:
            r = get_drop_row(state.board, action)
            if r >= 0:
                state.board[r, action] = opponent
                win = fast_win_check(state.board, opponent, r, action)
                state.board[r, action] = 0
                if win: return action
                
        # 3. Detectar movimientos suicidas (donde al tirar en 'x', provocamos que el rival gane en 'x' en el siguiente turno porque la ficha queda más alta)
        safe_actions = []
        for action in available:
            r = get_drop_row(state.board, action)
            if r > 0:
                state.board[r, action] = state.player
                state.board[r-1, action] = opponent
                win = fast_win_check(state.board, opponent, r-1, action)
                state.board[r-1, action] = 0
                state.board[r, action] = 0
                if win: continue # Es suicida, ignorarlo
            safe_actions.append(action)
            
        # Si solo queda 1 movimiento que no sea suicida, hacerlo y ya.
        if len(safe_actions) == 1:
            return safe_actions[0]
        # Si todos son suicidas (mala suerte), tirar en el primero por defecto
        if len(safe_actions) == 0:
            return available[0]
                
        return None

    def _minimax_alpha_beta(self, board, current_player, my_player, depth, alpha, beta):
        """
        Algoritmo Minimax optimizado con Poda Alpha-Beta.
        Se activa solo al final del juego para encontrar victorias forzadas evitando calcular el árbol de promedios probabilísticos.
        """
        available = [c for c in range(7) if board[0, c] == 0]
        
        if not available or depth == 0:
            return 0, None  # Empate o profundidad máxima (llegamos al fondo)
            
        # Ordenar movimientos (centro primero) para mayor probabilidad de poda (pruning).
        # Esto hace que el algoritmo Alpha-Beta se vuelva drásticamente más rápido.
        available.sort(key=lambda x: abs(x - 3))
        
        is_maximizing = (current_player == my_player)
        best_action = available[0]
        
        # Turno del Agente (Buscando maximizar la puntuación)
        if is_maximizing:
            max_eval = -float('inf')
            for action in available:
                r = get_drop_row(board, action)
                board[r, action] = current_player
                
                # Si ganamos con este movimiento, devolvemos un puntaje muy alto (1000) sumado con la profundidad.
                # Sumar la profundidad asegura que el algoritmo prefiera las victorias más "rápidas" posibles.
                if fast_win_check(board, current_player, r, action):
                    board[r, action] = 0
                    return 1000 + depth, action
                
                # Evaluar recursivamente el siguiente turno (el cual será del oponente: -current_player)
                eval_score, _ = self._minimax_alpha_beta(board, -current_player, my_player, depth-1, alpha, beta)
                board[r, action] = 0 # Deshacer el movimiento temporal
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                    
                # Alpha-Beta Pruning: Actualizar 'alpha'. Si alpha supera o iguala a 'beta', 
                # sabemos que el oponente jamás permitiría llegar hasta aquí, por lo que podemos podar ("cortar") esta rama del árbol.
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  
            return max_eval, best_action
            
        # Turno del Oponente (Buscando minimizar la puntuación del agente)
        else:
            min_eval = float('inf')
            for action in available:
                r = get_drop_row(board, action)
                board[r, action] = current_player
                
                # Si el oponente gana con este movimiento, le damos el mayor castigo posible (-1000) restando la profundidad.
                # Restar la profundidad hace que el agente elija perder "lo más tarde posible" si la derrota ya es forzada.
                if fast_win_check(board, current_player, r, action):
                    board[r, action] = 0
                    return -1000 - depth, action
                
                # Evaluar recursivamente el siguiente turno (que vuelve a ser el del agente)
                eval_score, _ = self._minimax_alpha_beta(board, -current_player, my_player, depth-1, alpha, beta)
                board[r, action] = 0 # Deshacer
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                    
                # Alpha-Beta Pruning: Actualizar 'beta' y cortar la rama si 'beta' baja demasiado
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_action
