import time
import random
from config import *

# A map of piece values using absolute numbers for easier calculations.
PIECE_SCORE_MAP = {
    'P': 20, 'N': 70, 'B': 70, 'K': 600
}

class B23CM1025:
    """
    An elite, high-speed winning agent using a fixed-depth search for maximum performance.
    - Fixed-Depth Search (Depth 5) for consistent, lightning-fast moves.
    - Transposition Tables to optimize the search.
    - Quiescence Search for tactical stability.
    - MVV-LVA move ordering.
    - A lean, powerful evaluation function focused on winning heuristics.
    """
    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 5  # A fixed depth for speed and strength.
        self.transposition_table = {}

    def get_best_move(self):
        """
        The main control function. Calls the minimax search with a fixed depth.
        """
        self.nodes_expanded = 0
        self.transposition_table.clear()
        
        try:
            score, best_move_found = self._minimax(self.depth, -float('inf'), float('inf'))
        except Exception:
            best_move_found = None
                
        # Failsafe in case of an error during search.
        if best_move_found is None:
            legal_moves = self.engine.get_legal_moves()
            if legal_moves:
                return random.choice(legal_moves)
            return None
                
        return best_move_found

    def _minimax(self, depth, alpha, beta):
        """
        The core alpha-beta search function with transposition table.
        """
        game_state = self.engine.get_game_state()
        if game_state != "ongoing":
            if game_state == "checkmate": return -float('inf'), None
            return 0, None

        board_key = (tuple(map(tuple, self.engine.board)), self.engine.white_to_move)
        if board_key in self.transposition_table and self.transposition_table[board_key]['depth'] >= depth:
            entry = self.transposition_table[board_key]
            if entry['score'] >= beta: return beta, entry['move']
            if entry['score'] <= alpha: return alpha, entry['move']
            alpha = max(alpha, entry['score'])

        if depth == 0:
            return self._quiescence_search(alpha, beta), None

        self.nodes_expanded += 1
        
        legal_moves = self.engine.get_legal_moves()
        legal_moves.sort(key=self._score_move, reverse=True)
        
        best_move = legal_moves[0] if legal_moves else None
        for move in legal_moves:
            self.engine.make_move(move)
            score = -self._minimax(depth - 1, -beta, -alpha)[0]
            self.engine.undo_move()
            
            if score > alpha:
                alpha = score
                best_move = move
            
            if alpha >= beta:
                break
        
        self.transposition_table[board_key] = {'score': alpha, 'move': best_move, 'depth': depth}
        
        return alpha, best_move

    def _score_move(self, move):
        """Helper function for MVV-LVA move ordering."""
        if move.piece_captured != EMPTY_SQUARE:
            victim_value = PIECE_SCORE_MAP.get(move.piece_captured[1], 0)
            attacker_value = PIECE_SCORE_MAP.get(move.piece_moved[1], 0)
            return victim_value - attacker_value + 1000
        return 0

    def _quiescence_search(self, alpha, beta):
        """A search for captures only, to ensure the evaluation is stable."""
        self.nodes_expanded += 1
        stand_pat_score = self.evaluate_board()

        if stand_pat_score >= beta: return beta
        if alpha < stand_pat_score: alpha = stand_pat_score

        capture_moves = [m for m in self.engine.get_legal_moves() if m.piece_captured != EMPTY_SQUARE]
        capture_moves.sort(key=self._score_move, reverse=True)

        for move in capture_moves:
            self.engine.make_move(move)
            score = -self._quiescence_search(-beta, -alpha)
            self.engine.undo_move()

            if score >= beta: return beta
            if score > alpha: alpha = score
        
        return alpha
    
    def evaluate_board(self):
        """A fast and powerful evaluation function."""
        score = 0
        center_squares = [(3, 1), (3, 2), (4, 1), (4, 2)]

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece != EMPTY_SQUARE:
                    is_white = piece[0] == 'w'
                    
                    score += PIECE_VALUES.get(piece, 0)
                    
                    piece_type = piece[1]
                    row_for_pst = r if is_white else (BOARD_HEIGHT - 1 - r)
                    
                    if piece_type == 'P': 
                        score += PAWN_PST[row_for_pst][c] if is_white else -PAWN_PST[row_for_pst][c]
                        # Fast Passed Pawn Check
                        is_passed = True
                        if is_white:
                            for i in range(r - 1, -1, -1):
                                if self.engine.board[i][c] == BLACK_PAWN: is_passed = False; break
                        else: # is_black
                            for i in range(r + 1, BOARD_HEIGHT):
                                if self.engine.board[i][c] == WHITE_PAWN: is_passed = False; break
                        if is_passed:
                            score += (7 - r) * 10 if is_white else - (r * 10)

                    elif piece_type == 'N': score += KNIGHT_PST[row_for_pst][c] if is_white else -KNIGHT_PST[row_for_pst][c]
                    elif piece_type == 'B': score += BISHOP_PST[row_for_pst][c] if is_white else -BISHOP_PST[row_for_pst][c]

                    if (r,c) in center_squares and piece_type in ('P', 'N'):
                        score += 5 if is_white else -5

        if self.engine.is_in_check():
            score += 2 if self.engine.white_to_move else -2
        return score if self.engine.white_to_move else-score