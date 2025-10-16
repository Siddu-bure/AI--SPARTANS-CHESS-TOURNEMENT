import sys
import copy
import time
from config import *
from board import GameEngine, Move

class B23CM1012:
    """
    Enhanced AI Player for 4x8 chess with improved evaluation and strategy
    """
    def __init__(self, board):
        self.engine = board
        self.nodes_expanded = 0
        self.depth = 4  # Optimal depth for 4x8 chess
        
    def get_best_move(self):
        """
        Calculates and returns the best move for the current board state.
        """
        self.nodes_expanded = 0
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        # Get all legal moves and order them
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None
            
        # If only one move available, return it immediately
        if len(legal_moves) == 1:
            return legal_moves[0]
            
        # Order moves for better alpha-beta pruning
        ordered_moves = self._order_moves(legal_moves)
            
        # Iterative deepening with time management
        start_time = time.time()
        max_depth = self.depth
        
        for depth in range(1, max_depth + 1):
            current_best = None
            current_alpha = alpha
            
            for move in ordered_moves:
                # Make the move
                self.engine.make_move(move)
                
                # Evaluate the position
                score = self._minimax(depth - 1, alpha, beta, False)
                self.nodes_expanded += 1
                
                # Undo the move
                self.engine.undo_move()
                
                if score > current_alpha:
                    current_alpha = score
                    current_best = move
                    
                # Alpha-beta pruning
                alpha = max(alpha, current_alpha)
                if alpha >= beta:
                    break
                    
            # Update best move for this depth
            if current_best:
                best_move = current_best
                
            # Check if we're running out of time
            if time.time() - start_time > 0.8:
                break
                
        return best_move
        
    def _order_moves(self, moves):
        """
        Order moves to improve alpha-beta pruning efficiency.
        """
        scored_moves = []
        
        for move in moves:
            score = 0
            
            # Prioritize captures by MVV-LVA
            if move.piece_captured != EMPTY_SQUARE:
                aggressor_value = abs(PIECE_VALUES.get(move.piece_moved, 0))
                victim_value = abs(PIECE_VALUES.get(move.piece_captured, 0))
                
                # MVV-LVA with bonus for good exchanges
                score = victim_value * 10 - aggressor_value + 1000
                
                # Penalize bad exchanges more severely
                if victim_value < aggressor_value:
                    score -= 800  # Strong penalty for losing material
                elif victim_value > aggressor_value:
                    score += 300   # Bonus for winning material
                    
            # Prioritize checks with higher bonus
            self.engine.make_move(move)
            if self.engine.is_in_check():
                score += 300  # Increased check bonus
            self.engine.undo_move()
            
            # Prioritize center moves in 4x8 board
            end_col = move.end_col
            if end_col in [1, 2]:  # Center files in 4x8 board
                score += 50
                
            # Prioritize developing knights and bishops early
            if move.piece_moved in [WHITE_KNIGHT, BLACK_KNIGHT, WHITE_BISHOP, BLACK_BISHOP]:
                score += 30
                
            scored_moves.append((score, move))
        
        # Sort moves by score (highest first)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]
        
    def _minimax(self, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning.
        """
        # Base case: reached maximum depth
        if depth == 0:
            return self._quiescence_search(alpha, beta, maximizing_player)
            
        # Get legal moves and order them
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            # Checkmate or stalemate
            if self.engine.is_in_check():
                return float('-inf') if maximizing_player else float('inf')
            return 0  # Stalemate
            
        # Order moves for better alpha-beta pruning
        ordered_moves = self._order_moves(legal_moves)
            
        if maximizing_player:
            max_eval = float('-inf')
            for move in ordered_moves:
                self.engine.make_move(move)
                eval_score = self._minimax(depth - 1, alpha, beta, False)
                self.engine.undo_move()
                self.nodes_expanded += 1
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                self.engine.make_move(move)
                eval_score = self._minimax(depth - 1, alpha, beta, True)
                self.engine.undo_move()
                self.nodes_expanded += 1
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
            
    def _quiescence_search(self, alpha, beta, maximizing_player):
        """
        Quiescence search to avoid horizon effect.
        """
        stand_pat = self.evaluate_board()
        
        if maximizing_player:
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if beta > stand_pat:
                beta = stand_pat
                
        # Get all captures and checks
        tactical_moves = []
        for move in self.engine.get_legal_moves():
            if move.piece_captured != EMPTY_SQUARE or self._gives_check(move):
                tactical_moves.append(move)
                
        # Order tactical moves
        tactical_moves = self._order_moves(tactical_moves)
        
        if maximizing_player:
            for move in tactical_moves:
                self.engine.make_move(move)
                score = self._quiescence_search(alpha, beta, False)
                self.engine.undo_move()
                self.nodes_expanded += 1
                
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            return alpha
        else:
            for move in tactical_moves:
                self.engine.make_move(move)
                score = self._quiescence_search(alpha, beta, True)
                self.engine.undo_move()
                self.nodes_expanded += 1
                
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score
            return beta
            
    def _gives_check(self, move):
        """
        Check if a move gives check.
        """
        self.engine.make_move(move)
        gives_check = self.engine.is_in_check()
        self.engine.undo_move()
        return gives_check
            
    def evaluate_board(self):
        """
        Enhanced evaluation function for 4x8 chess.
        """
        # Material value
        material = 0
        positional = 0
        
        # Count pieces and evaluate position
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece != EMPTY_SQUARE:
                    # Material value
                    material += PIECE_VALUES.get(piece, 0)
                    
                    # Positional value based on piece-square tables
                    if piece == WHITE_PAWN:
                        positional += PAWN_PST[r][c]
                        # Bonus for advanced pawns in 4x8 chess
                        if r <= 3:  # Advanced pawns
                            positional += (7 - r) * 5
                    elif piece == BLACK_PAWN:
                        positional -= PAWN_PST[r][c]
                        if r >= 4:  # Advanced pawns
                            positional -= r * 5
                    elif piece == WHITE_KNIGHT:
                        positional += KNIGHT_PST[r][c]
                        # Knights are more valuable in center in 4x8
                        if c in [1, 2]:
                            positional += 10
                    elif piece == BLACK_KNIGHT:
                        positional -= KNIGHT_PST[r][c]
                        if c in [1, 2]:
                            positional -= 10
                    elif piece == WHITE_BISHOP:
                        positional += BISHOP_PST[r][c]
                        # Bishops on long diagonals are powerful
                        if r == c or (r + c) == 3:
                            positional += 15
                    elif piece == BLACK_BISHOP:
                        positional -= BISHOP_PST[r][c]
                        if r == c or (r + c) == 3:
                            positional -= 15
                    elif piece == WHITE_KING:
                        # King safety is crucial in 4x8 chess
                        positional += KING_PST_LATE_GAME[r][c]
                        if r >= 6:  # King should stay back
                            positional += 20
                    elif piece == BLACK_KING:
                        positional -= KING_PST_LATE_GAME[r][c]
                        if r <= 1:  # King should stay back
                            positional -= 20
        
        # Check and checkmate threats
        game_state = self.engine.get_game_state()
        if game_state == "checkmate":
            if self.engine.white_to_move:
                return float('-inf') + 1000  # Ensure proper ordering
            else:
                return float('inf') - 1000
                
        # Check bonus - increased for 4x8 chess
        check_bonus = 0
        if self.engine.is_in_check():
            check_bonus = 80 if self.engine.white_to_move else -80
            
        # Center control bonus - more important in 4x8
        center_control = 0
        center_squares = [(3, 1), (3, 2), (4, 1), (4, 2)]
        for r, c in center_squares:
            piece = self.engine.board[r][c]
            if piece.startswith('w'):
                center_control += 15
            elif piece.startswith('b'):
                center_control -= 15
                
        # Mobility bonus - more important in cramped 4x8 board
        mobility = len(self.engine.get_legal_moves())
        mobility_bonus = mobility * 3 if self.engine.white_to_move else -mobility * 3
        
        # King safety - crucial in 4x8 chess
        king_safety = self._evaluate_king_safety()
            
        # Pawn structure evaluation
        pawn_structure = self._evaluate_pawn_structure()
        
        # Piece activity bonus
        piece_activity = self._evaluate_piece_activity()
            
        # Total evaluation with adjusted weights for 4x8 chess
        total_score = (material * 1.2 + positional + check_bonus + center_control * 1.5 + mobility_bonus + king_safety * 2 + pawn_structure + piece_activity)
        
        # Adjust perspective based on whose turn it is
        if not self.engine.white_to_move:
            total_score = -total_score
            
        return total_score
        
    def _evaluate_king_safety(self):
        """
        Evaluate king safety - crucial in 4x8 chess.
        """
        safety = 0
        white_king_pos = self.engine._find_king('w')
        black_king_pos = self.engine._find_king('b')
        
        # Penalize exposed kings
        if white_king_pos:
            r, c = white_king_pos
            # King should stay on the back rank in 4x8
            if r < 6:
                safety -= (6 - r) * 20
            # King should avoid the edges
            if c == 0 or c == 3:
                safety -= 15
                
        if black_king_pos:
            r, c = black_king_pos
            if r > 1:
                safety += (r - 1) * 20
            if c == 0 or c == 3:
                safety += 15
                
        return safety
        
    def _evaluate_pawn_structure(self):
        """
        Evaluate pawn structure for 4x8 chess.
        """
        white_pawns = [0] * BOARD_WIDTH
        black_pawns = [0] * BOARD_WIDTH
        
        doubled_penalty = 0
        isolated_penalty = 0
        passed_bonus = 0
        
        # Count pawns and find passed pawns
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece == WHITE_PAWN:
                    white_pawns[c] += 1
                    # Check if passed pawn
                    is_passed = True
                    for rr in range(r-1, -1, -1):
                        if self.engine.board[rr][c] == BLACK_PAWN:
                            is_passed = False
                            break
                        if c > 0 and self.engine.board[rr][c-1] == BLACK_PAWN:
                            is_passed = False
                            break
                        if c < 3 and self.engine.board[rr][c+1] == BLACK_PAWN:
                            is_passed = False
                            break
                    if is_passed:
                        passed_bonus += (7 - r) * 20  # Bonus based on advancement
                        
                elif piece == BLACK_PAWN:
                    black_pawns[c] += 1
                    # Check if passed pawn
                    is_passed = True
                    for rr in range(r+1, BOARD_HEIGHT):
                        if self.engine.board[rr][c] == WHITE_PAWN:
                            is_passed = False
                            break
                        if c > 0 and self.engine.board[rr][c-1] == WHITE_PAWN:
                            is_passed = False
                            break
                        if c < 3 and self.engine.board[rr][c+1] == WHITE_PAWN:
                            is_passed = False
                            break
                    if is_passed:
                        passed_bonus -= r * 20  # Bonus based on advancement
        
        # Evaluate doubled pawns
        for c in range(BOARD_WIDTH):
            if white_pawns[c] > 1:
                doubled_penalty -= 20 * (white_pawns[c] - 1)
            if black_pawns[c] > 1:
                doubled_penalty += 20 * (black_pawns[c] - 1)
                
        # Evaluate isolated pawns
        for c in range(BOARD_WIDTH):
            is_white_isolated = (white_pawns[c] > 0 and 
                                (c == 0 or white_pawns[c-1] == 0) and 
                                (c == BOARD_WIDTH-1 or white_pawns[c+1] == 0))
            is_black_isolated = (black_pawns[c] > 0 and 
                                (c == 0 or black_pawns[c-1] == 0) and 
                                (c == BOARD_WIDTH-1 or black_pawns[c+1] == 0))
            
            if is_white_isolated:
                isolated_penalty -= 25
            if is_black_isolated:
                isolated_penalty += 25
                
        return doubled_penalty + isolated_penalty + passed_bonus
        
    def _evaluate_piece_activity(self):
        """
        Evaluate piece activity - knights and bishops.
        """
        activity = 0
        
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                
                if piece == WHITE_KNIGHT:
                    # Knights are more active when not on the edge
                    if c not in [0, 3]:
                        activity += 10
                    # Knights controlling center
                    if (r in [2, 3, 4, 5] and c in [1, 2]):
                        activity += 15
                        
                elif piece == BLACK_KNIGHT:
                    if c not in [0, 3]:
                        activity -= 10
                    if (r in [2, 3, 4, 5] and c in [1, 2]):
                        activity -= 15
                        
                elif piece == WHITE_BISHOP:
                    # Bishops on long diagonals
                    if (r + c) % 2 == 0:  # Same color squares
                        activity += 10
                    # Bishops controlling center
                    if (r in [2, 3, 4, 5] and c in [1, 2]):
                        activity += 15
                        
                elif piece == BLACK_BISHOP:
                    if (r + c) % 2 == 0:
                        activity -= 10
                    if (r in [2, 3, 4, 5] and c in [1, 2]):
                        activity -= 15
        return activity