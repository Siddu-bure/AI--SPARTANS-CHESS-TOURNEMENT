# Global definitions for pieces and board dimensions
BOARD_WIDTH = 4
BOARD_HEIGHT = 8

# Piece representation using single characters
WHITE_PAWN = 'wP'
BLACK_PAWN = 'bP'
WHITE_KNIGHT = 'wN'
BLACK_KNIGHT = 'bN'
WHITE_BISHOP = 'wB'
BLACK_BISHOP = 'bB'
WHITE_KING = 'wK'
BLACK_KING = 'bK'
EMPTY_SQUARE = '--'

# Defines the total material value (excluding kings) below which the endgame logic activates.
ENDGAME_MATERIAL_THRESHOLD = 300


# Piece values updated to reflect the assignment's scoring rules.
PIECE_VALUES = {
    WHITE_PAWN: 20,
    BLACK_PAWN: -20,
    WHITE_KNIGHT: 70,
    BLACK_KNIGHT: -70,
    WHITE_BISHOP: 70,
    BLACK_BISHOP: -70,
    WHITE_KING: 600,
    BLACK_KING: -600
}

# Piece-Square Tables (PST) for positional evaluation.
PAWN_PST = [
    [0, 0, 0, 0],
    [50, 50, 50, 50],
    [40, 40, 40, 40],
    [30, 30, 30, 30],
    [20, 20, 20, 20],
    [10, 10, 10, 10],
    [5, 5, 5, 5],
    [0, 0, 0, 0],
]

KNIGHT_PST = [
    [-10, -5, -5, -10],
    [-5,  0,  0, -5],
    [-5,  5,  5, -5],
    [ 0, 10, 10,  0],
    [ 0, 10, 10,  0],
    [-5,  5,  5, -5],
    [-5,  0,  0, -5],
    [-10, -5, -5, -10],
]

# --- MODIFICATION START ---
# Standard bishop PST for the middlegame.
BISHOP_PST_MIDDLEGAME = [
    [-5, -5, -5, -5],
    [-5, 10, 10, -5],
    [-5, 10, 10, -5],
    [ 0, 10, 10,  0],
    [ 0, 10, 10,  0],
    [-5, 10, 10, -5],
    [-5, 10, 10, -5],
    [-5, -5, -5, -5],
]

# In the endgame, bishops are powerful. This table heavily penalizes passive bishops
# on the back rank and rewards active, central bishops.
BISHOP_PST_END_GAME = [
    [-20, -10, -10, -20], # Rank 8 (Very bad for white's bishop)
    [-10,   5,   5, -10],
    [ -5,  10,  10,  -5],
    [  0,  15,  15,   0],
    [  0,  15,  15,   0],
    [ -5,  10,  10,  -5],
    [-10,   5,   5, -10],
    [-20, -10, -10, -20], # Rank 1 (Very bad for white's bishop)
]
# --- MODIFICATION END ---


# King safety is paramount in the middlegame.
KING_PST_MIDDLEGAME = [
    [ 20, 30, 10,  0],
    [ 20, 20,  0,  0],
    [-10,-20,-20,-10],
    [-20,-30,-30,-20],
    [-20,-30,-30,-20],
    [-10,-20,-20,-10],
    [ 20, 20,  0,  0],
    [ 20, 30, 10,  0]
]

# In the endgame, the king becomes an attacking piece.
KING_PST_END_GAME = [
    [-10, -5, -5, -10],
    [ -5,  5,  5, -5],
    [  0, 10, 10,  0],
    [  5, 20, 20,  5],
    [  5, 20, 20,  5],
    [  0, 10, 10,  0],
    [ -5,  5,  5, -5],
    [-10, -5, -5, -10],
]

PIECE_SYMBOLS = {
    'wP': '♙', 'bP': '♟', 'wN': '♘', 'bN': '♞',
    'wB': '♗', 'bB': '♝', 'wK': '♔', 'bK': '♚',
    '--': ' '
}