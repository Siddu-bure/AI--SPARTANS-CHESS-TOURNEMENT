import random
import time
from board import Move
from config import *

class B23CM1011:
    
    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 8  # Max search depth
        self.tt = {}    # Transposition table
        self.history = {}  # History heuristic
        self.moves_played = 0

   
    # Public entry
   
    def get_best_move(self):
        start_time = time.time()
        self.nodes_expanded = 0
        self.tt = {}
        best_move = None

        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None

        self.moves_played += 1
        if self.moves_played < 15:
            time_limit = 0.3
        else:
            time_limit = 0.9

        for depth in range(1, self.depth + 1):
            if time.time() - start_time > time_limit:
                break
            try:
                search_time_limit = time_limit - (time.time() - start_time)
                if search_time_limit < 0.1: break
                move, _ = self.minimax(depth, -float('inf'), float('inf'), self.engine.white_to_move, time.time(), search_time_limit - 0.05)
                if move:
                    best_move = move
            except TimeoutError:
                break

        if not best_move:
            best_move = random.choice(legal_moves)

        return best_move

   
    # Minimax with alpha-beta
    
    def minimax(self, depth, alpha, beta, maximizing_player, start_time, time_limit):
        if time.time() - start_time > time_limit:
            raise TimeoutError()

        board_hash = (tuple(map(tuple, self.engine.board)), self.engine.white_to_move)
        tt_entry = self.tt.get(board_hash)

        if tt_entry and tt_entry['depth'] >= depth:
            return tt_entry['move'], tt_entry['score']

        self.nodes_expanded += 1

        if depth == 0:
            return None, self.quiescence_search(alpha, beta, start_time, time_limit)

        legal_moves = self.engine.get_legal_moves()
        game_state = self.engine.get_game_state()
        if not legal_moves or game_state != "ongoing":
            return None, self.evaluate_board()

        ordered_moves = self._order_moves(legal_moves, tt_entry_move=(tt_entry['move'] if tt_entry else None))
        best_move = ordered_moves[0]

        if maximizing_player:
            max_eval = -float('inf')
            for move in ordered_moves:
                self.engine.make_move(move)
                try:
                    _, current_eval = self.minimax(depth - 1, alpha, beta, False, start_time, time_limit)
                finally:
                    self.engine.undo_move()

                if current_eval > max_eval:
                    max_eval = current_eval
                    best_move = move
                alpha = max(alpha, current_eval)
                if beta <= alpha:
                    break
            self.tt[board_hash] = {'depth': depth, 'score': max_eval, 'move': best_move}
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                self.engine.make_move(move)
                try:
                    _, current_eval = self.minimax(depth - 1, alpha, beta, True, start_time, time_limit)
                finally:
                    self.engine.undo_move()

                if current_eval < min_eval:
                    min_eval = current_eval
                    best_move = move
                beta = min(beta, current_eval)
                if beta <= alpha:
                    break
            self.tt[board_hash] = {'depth': depth, 'score': min_eval, 'move': best_move}
            return best_move, min_eval

    
    # Quiescence search
    
    def quiescence_search(self, alpha, beta, start_time, time_limit, depth=2):
        if time.time() - start_time > time_limit:
            raise TimeoutError()

        self.nodes_expanded += 1
        stand_pat = self.evaluate_board()
        if depth == 0:
            return stand_pat

        if self.engine.white_to_move:
            if stand_pat >= beta: return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha: return alpha
            beta = min(beta, stand_pat)

        capture_moves = [m for m in self.engine.get_legal_moves() if m.piece_captured != EMPTY_SQUARE]
        ordered_captures = self._order_moves(capture_moves)

        for move in ordered_captures:
            self.engine.make_move(move)
            try:
                score = self.quiescence_search(alpha, beta, start_time, time_limit, depth - 1)
            finally:
                self.engine.undo_move()

            if self.engine.white_to_move:
                alpha = max(alpha, score)
                if alpha >= beta: return beta
            else:
                beta = min(beta, score)
                if beta <= alpha: return alpha

        return alpha if self.engine.white_to_move else beta

    # Move ordering with safe TT move
    
    def _order_moves(self, moves, tt_entry_move=None):
        center = {(3,3), (3,4), (4,3), (4,4)}

        def score_move(m):
            s = 0

            # TT move
            if tt_entry_move:
                if isinstance(tt_entry_move, Move):
                    if self._same_move(m, tt_entry_move):
                        s += 10000
                elif isinstance(tt_entry_move, tuple):
                    if self._same_move(m, tt_entry_move):
                        s += 10000

            # Captures
            if m.piece_captured != EMPTY_SQUARE:
                victim = PIECE_VALUES.get(m.piece_captured[1], 0)
                attacker = PIECE_VALUES.get(m.piece_moved[1], 0)
                s += victim * 100 - attacker * 10

            # Checks
            self.engine.make_move(m)
            if self.engine.is_in_check():
                s += 500
            self.engine.undo_move()

            # Center
            if (m.end_row, m.end_col) in center:
                s += 20

            # History heuristic
            key_move = (m.start_row, m.start_col, m.end_row, m.end_col)
            s += self.history.get(key_move, 0)

            return s

        return sorted(moves, key=score_move, reverse=True)

    def _same_move(self, m1, m2):
        if m1 is None or m2 is None:
            return False
        if isinstance(m2, Move):
            return (m1.start_row == m2.start_row and m1.start_col == m2.start_col and
                    m1.end_row == m2.end_row and m1.end_col == m2.end_col and
                    m1.piece_moved == m2.piece_moved)
        elif isinstance(m2, tuple):
            return (m1.start_row == m2[0] and m1.start_col == m2[1] and
                    m1.end_row == m2[2] and m1.end_col == m2[3])
        return False

    
    # Board evaluation
    
    def evaluate_board(self):
        gs = self.engine.get_game_state()
        if gs == "checkmate":
            return 600 if not self.engine.white_to_move else -600
        if gs == "stalemate":
            return 0

        score = 0
        total_pieces = 0
        white_king_pos, black_king_pos = None, None

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece != EMPTY_SQUARE:
                    total_pieces += 1
                    if piece == WHITE_KING: white_king_pos = (r,c)
                    elif piece == BLACK_KING: black_king_pos = (r,c)

                    score += PIECE_VALUES.get(piece, 0)
                    piece_type, color_multiplier = piece[1], 1 if piece[0] == 'w' else -1
                    if piece_type == 'P': score += PAWN_PST[r][c] * color_multiplier
                    elif piece_type == 'N': score += KNIGHT_PST[r][c] * color_multiplier
                    elif piece_type == 'B': score += BISHOP_PST[r][c] * color_multiplier
                    elif piece_type == 'K': score += KING_PST_LATE_GAME[r][c] * color_multiplier

        if self.engine.get_repetition_count() >= 2: 
            return 0

        if total_pieces <= 8 and white_king_pos and black_king_pos:
            king_dist = abs(white_king_pos[0] - black_king_pos[0]) + abs(white_king_pos[1] - black_king_pos[1])
            if score > 100: score += (10 - king_dist) * 10
            elif score < -100: score -= (10 - king_dist) * 10
            op_king_pos = black_king_pos if score > 0 else white_king_pos
            center_dist_r = abs(op_king_pos[0] - 3.5)
            center_dist_c = abs(op_king_pos[1] - 1.5)
            edge_bonus = (center_dist_r + center_dist_c) * 8
            score += edge_bonus if score > 0 else -edge_bonus

        if self.engine.is_in_check():
            score += -50 if self.engine.white_to_move else 50

        return score


class TimeoutError(Exception):
    pass
