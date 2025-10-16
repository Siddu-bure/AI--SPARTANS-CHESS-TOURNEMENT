import time
from config_ import *
#from board import GameEngine

class MyAIPlayer:
    """
    An AI player using Iterative Deepening, Alpha-Beta Pruning, a Transposition Table,
    advanced Move Ordering, and Quiescence Search to mitigate the horizon effect.
    """
    def __init__(self, board_engine):
        self.engine = board_engine
        self.nodes_expanded = 0
        self.depth = 4  # Max search depth for middlegame
        self.endgame_depth = 5 # Increased depth for endgame analysis
        self.time_limit_per_move = 60
        self.transposition_table = {}

    def get_best_move(self):
        start_time = time.time()
        legal_moves = self.engine.get_legal_moves()

        if not legal_moves:
            return None
        
        # Careful "Freebie" Capture: Before any deep search, check for captures of undefended pieces.
        best_freebie_capture = None
        max_freebie_value = 0
        
        capture_moves = [m for m in legal_moves if m.piece_captured != EMPTY_SQUARE]

        for move in capture_moves:
            # Temporarily make the move to see the outcome
            self.engine.make_move(move)
            # Check if the opponent can recapture on the destination square
            is_recapturable = self.engine._is_square_attacked((move.end_row, move.end_col), move.piece_moved[0])
            self.engine.undo_move()

            # If it's not recapturable, it's a "free" piece
            if not is_recapturable:
                captured_value = abs(PIECE_VALUES.get(move.piece_captured, 0))
                if captured_value > max_freebie_value:
                    max_freebie_value = captured_value
                    best_freebie_capture = move
        
        if best_freebie_capture:
            return best_freebie_capture
        
        self.transposition_table.clear()
        best_move_found = legal_moves[0]

        # Determine search depth based on game phase
        w_material = 0
        b_material = 0
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece != EMPTY_SQUARE and piece[1] != 'K':
                     if piece.startswith('w'):
                         w_material += abs(PIECE_VALUES.get(piece, 0))
                     else:
                         b_material += abs(PIECE_VALUES.get(piece, 0))
        
        is_endgame = (w_material + b_material) < ENDGAME_MATERIAL_THRESHOLD
        max_depth_for_turn = self.endgame_depth if is_endgame else self.depth
        
        try:
            for current_depth in range(1, max_depth_for_turn + 1):
                if time.time() - start_time >= self.time_limit_per_move:
                    break
                
                best_move_this_iteration, _ = self._search(current_depth, start_time)
                
                if time.time() - start_time < self.time_limit_per_move and best_move_this_iteration:
                    best_move_found = best_move_this_iteration
                else:
                    break
        except TimeoutError:
            pass

        return best_move_found

    def _score_move(self, move):
        """Scores a move for ordering purposes to improve alpha-beta pruning."""
        score = 0
        if move.piece_captured != EMPTY_SQUARE:
            victim_value = abs(PIECE_VALUES.get(move.piece_captured, 0))
            aggressor_value = abs(PIECE_VALUES.get(move.piece_moved, 0))
            score += 1000 + victim_value - (aggressor_value / 10)

        self.engine.make_move(move)
        
        if self.engine.is_in_check():
            score += 500
        
        # This penalty is for move ordering only; the main evaluation handles repetition logic
        if self.engine.get_repetition_count() >= 1:
            score -= 250

        piece_color = move.piece_moved[0]
        is_attacked = self.engine._is_square_attacked((move.end_row, move.end_col), piece_color)
        
        self.engine.undo_move()

        if move.piece_moved[1] == 'P':
            score -= 10
        
        if is_attacked:
            opponent_color = 'b' if piece_color == 'w' else 'w'
            is_defended = self.engine._is_square_attacked((move.end_row, move.end_col), opponent_color)
            if not is_defended:
                score -= abs(PIECE_VALUES[move.piece_moved]) * 2
        
        return score

    def _sort_moves(self, moves, captures_only=False):
        """Sorts moves based on their score for more efficient pruning."""
        if captures_only:
             moves = [m for m in moves if m.piece_captured != EMPTY_SQUARE]
        moves.sort(key=lambda m: self._score_move(m), reverse=True)
        return moves

    def _search(self, depth, start_time):
        self.nodes_expanded = 0
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        
        moves = self.engine.get_legal_moves()
        
        board_key = (tuple(map(tuple, self.engine.board)), self.engine.white_to_move)
        prev_best_move = self.transposition_table.get(board_key, {}).get('best_move')
        if prev_best_move and prev_best_move in moves:
            moves.remove(prev_best_move)
            moves.insert(0, prev_best_move)
        
        moves = self._sort_moves(moves)

        best_score = -float('inf')
        for move in moves:
            self.engine.make_move(move)
            score = -self._alpha_beta(depth - 1, -beta, -alpha, start_time)
            self.engine.undo_move()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        
        if best_move is not None:
             self.transposition_table[board_key] = {
                'score': best_score, 'depth': depth, 'flag': 'EXACT', 'best_move': best_move
            }

        return best_move, best_score

    def _alpha_beta(self, depth, alpha, beta, start_time):
        self.nodes_expanded += 1
        if time.time() - start_time >= self.time_limit_per_move:
            raise TimeoutError()
        
        original_alpha = alpha
        board_key = (tuple(map(tuple, self.engine.board)), self.engine.white_to_move)

        if board_key in self.transposition_table and self.transposition_table[board_key]['depth'] >= depth:
            entry = self.transposition_table[board_key]
            if entry['flag'] == 'EXACT': return entry['score']
            elif entry['flag'] == 'LOWERBOUND': alpha = max(alpha, entry['score'])
            elif entry['flag'] == 'UPPERBOUND': beta = min(beta, entry['score'])
            if alpha >= beta: return entry['score']

        game_state = self.engine.get_game_state()
        if game_state != "ongoing":
            return self.evaluate_board()

        if depth == 0:
            return self._quiescence_search(alpha, beta, start_time)

        moves = self._sort_moves(self.engine.get_legal_moves())
        
        best_move = None
        for move in moves:
            self.engine.make_move(move)
            score = -self._alpha_beta(depth - 1, -beta, -alpha, start_time)
            self.engine.undo_move()

            if score > alpha:
                alpha = score
                best_move = move
            if alpha >= beta: break
    
        if alpha <= original_alpha: flag = 'UPPERBOUND'
        elif alpha >= beta: flag = 'LOWERBOUND'
        else: flag = 'EXACT'
        
        self.transposition_table[board_key] = {
            'score': alpha, 'depth': depth, 'flag': flag, 'best_move': best_move
        }
        
        return alpha
        
    def _quiescence_search(self, alpha, beta, start_time):
        self.nodes_expanded += 1
        
        stand_pat_score = self.evaluate_board()
        if stand_pat_score >= beta:
            return beta
        alpha = max(alpha, stand_pat_score)

        moves = self._sort_moves(self.engine.get_legal_moves(), captures_only=True)

        for move in moves:
            if time.time() - start_time >= self.time_limit_per_move:
                raise TimeoutError()

            self.engine.make_move(move)
            score = -self._quiescence_search(-beta, -alpha, start_time)
            self.engine.undo_move()

            if score >= beta:
                return beta
            alpha = max(alpha, score)

        return alpha

    def evaluate_board(self):
        game_state = self.engine.get_game_state()
        if game_state == "checkmate": return -100000 if self.engine.white_to_move else 100000
        if game_state == "stalemate" or self.engine.get_repetition_count() >= 3: return 0
        
        w_pieces = {'P': 0, 'N': 0, 'B': 0, 'K': 0}
        b_pieces = {'P': 0, 'N': 0, 'B': 0, 'K': 0}
        w_king_pos, b_king_pos = None, None

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece != EMPTY_SQUARE:
                    piece_type = piece[1]
                    if piece.startswith('w'):
                        w_pieces[piece_type] += 1
                        if piece_type == 'K': w_king_pos = (r, c)
                    else:
                        b_pieces[piece_type] += 1
                        if piece_type == 'K': b_king_pos = (r, c)

        w_material = w_pieces['P'] * 20 + w_pieces['N'] * 70 + w_pieces['B'] * 70
        b_material = b_pieces['P'] * 20 + b_pieces['N'] * 70 + b_pieces['B'] * 70
        
        total_material = w_material + b_material
        is_endgame = total_material < ENDGAME_MATERIAL_THRESHOLD
        
        material_score = 0
        positional_score = 0
        positional_weight = 0.5

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece == EMPTY_SQUARE: continue
                
                material_score += PIECE_VALUES.get(piece, 0)
                
                piece_type = piece[1]
                if piece_type == 'P': pst = PAWN_PST
                elif piece_type == 'N': pst = KNIGHT_PST
                elif piece_type == 'B':
                    pst = BISHOP_PST_END_GAME if is_endgame else BISHOP_PST_MIDDLEGAME
                elif piece_type == 'K':
                    pst = KING_PST_END_GAME if is_endgame else KING_PST_MIDDLEGAME
                else:
                    pst = []

                if pst:
                    if piece.startswith('w'):
                        positional_score += pst[r][c]
                    else:
                        positional_score -= pst[7 - r][c]

        # Specific checkmate endgame logic (e.g., KBN vs K)
        b_has_only_king = (b_material == 0)
        if is_endgame and w_pieces['B'] == 1 and w_pieces['N'] == 2 and b_has_only_king:
            mate_bonus = self._score_kbn_endgame('w', w_king_pos, b_king_pos)
            final_score = material_score + mate_bonus
            return final_score if self.engine.white_to_move else -final_score
        
        final_score = material_score + (positional_weight * positional_score)
        
        # General endgame logic for king activity
        if is_endgame and w_king_pos and b_king_pos:
            material_advantage = w_material - b_material
            king_dist = abs(w_king_pos[0] - b_king_pos[0]) + abs(w_king_pos[1] - b_king_pos[1])
            
            if material_advantage > 60: # White is winning
                final_score += (10 - king_dist) * 25
                b_king_r, b_king_c = b_king_pos
                dist_from_center = abs(b_king_r - 3.5) + abs(b_king_c - 1.5)
                final_score += dist_from_center * 15 # Push opponent's king to the edge
            
            elif material_advantage < -60: # Black is winning
                final_score -= (10 - king_dist) * 25
                w_king_r, w_king_c = w_king_pos
                dist_from_center = abs(w_king_r - 3.5) + abs(w_king_c - 1.5)
                final_score -= dist_from_center * 15 # Push opponent's king to the edge

        # Context-Aware Repetition Handling
        if self.engine.get_repetition_count() == 2:
            material_advantage = w_material - b_material
            # If we are winning, repeating a move is a blunder. Penalize it heavily.
            current_player_is_winning = (self.engine.white_to_move and material_advantage > 20) or \
                                      (not self.engine.white_to_move and material_advantage < -20)
            current_player_is_losing = (self.engine.white_to_move and material_advantage < -20) or \
                                     (not self.engine.white_to_move and material_advantage > 20)

            if current_player_is_winning:
                final_score -= 150 # Avoid draws when ahead
            elif current_player_is_losing:
                final_score += 150 # Seek draws when behind
            else:
                final_score -= 10 # Slightly discourage draws in equal positions

        return final_score if self.engine.white_to_move else -final_score

    def _score_kbn_endgame(self, winning_color, winning_king_pos, losing_king_pos):
        """
        Calculates a strong evaluation bonus for the Bishop and Two Knights mate.
        """
        if winning_king_pos is None or losing_king_pos is None: return 0

        bishop_pos = None
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece.startswith(winning_color) and piece[1] == 'B':
                    bishop_pos = (r, c)
                    break
            if bishop_pos: break
        
        if bishop_pos is None: return 0

        bishop_square_color = (bishop_pos[0] + bishop_pos[1]) % 2

        # Define the correct mating corners based on the bishop's color
        if bishop_square_color == 0: # Dark-squared bishop
            target_corners = [(0, 0), (7, 3)]
        else: # Light-squared bishop
            target_corners = [(0, 3), (7, 0)]
        
        # 1. Calculate the defending king's distance to the nearest correct corner
        lk_r, lk_c = losing_king_pos
        dist1 = abs(lk_r - target_corners[0][0]) + abs(lk_c - target_corners[0][1])
        dist2 = abs(lk_r - target_corners[1][0]) + abs(lk_c - target_corners[1][1])
        min_dist_to_corner = min(dist1, dist2)

        # 2. Calculate the distance between the two kings
        wk_r, wk_c = winning_king_pos
        king_dist = abs(wk_r - lk_r) + abs(wk_c - lk_c)
        
        # Create a massive bonus for cornering the king and a strong bonus for king proximity.
        cornering_bonus = (10 - min_dist_to_corner) * 50
        king_proximity_bonus = (10 - king_dist) * 25
        
        total_bonus = cornering_bonus + king_proximity_bonus
        return total_bonus if winning_color == 'w' else -total_bonus


class TimeoutError(Exception):
    pass