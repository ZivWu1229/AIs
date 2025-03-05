import random
import curses

def init_game_board():
    return [[0] * 4 for _ in range(4)]

def add_new_number(board):
    empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if empty_cells:
        r, c = random.choice(empty_cells)
        board[r][c] = 2 if random.random() < 0.9 else 4

def compress(board):
    new_board = [[0] * 4 for _ in range(4)]
    for r in range(4):
        pos = 0
        for c in range(4):
            if board[r][c] != 0:
                new_board[r][pos] = board[r][c]
                pos += 1
    return new_board

def merge(board):
    for r in range(4):
        for c in range(3):
            if board[r][c] == board[r][c + 1] and board[r][c] != 0:
                board[r][c] *= 2
                board[r][c + 1] = 0
    return board

def reverse(board):
    return [row[::-1] for row in board]

def transpose(board):
    return [list(row) for row in zip(*board)]

def move_left(board):
    board = compress(board)
    board = merge(board)
    board = compress(board)
    return board

def move_right(board):
    board = reverse(board)
    board = move_left(board)
    board = reverse(board)
    return board

def move_up(board):
    board = transpose(board)
    board = move_left(board)
    board = transpose(board)
    return board

def move_down(board):
    board = transpose(board)
    board = move_right(board)
    board = transpose(board)
    return board

def check_game_over(board):
    for r in range(4):
        for c in range(4):
            if board[r][c] == 0:
                return False
            if c < 3 and board[r][c] == board[r][c + 1]:
                return False
            if r < 3 and board[r][c] == board[r + 1][c]:
                return False
    return True

def draw_board(stdscr, board):
    stdscr.clear()
    for r in range(4):
        for c in range(4):
            stdscr.addstr(r * 2, c * 5, str(board[r][c]).center(5) if board[r][c] != 0 else "     ")
    stdscr.refresh()

def game_loop(stdscr):
    curses.curs_set(0)
    board = init_game_board()
    add_new_number(board)
    add_new_number(board)
    while True:
        draw_board(stdscr, board)
        key = stdscr.getch()
        if key == curses.KEY_LEFT:
            board = move_left(board)
        elif key == curses.KEY_RIGHT:
            board = move_right(board)
        elif key == curses.KEY_UP:
            board = move_up(board)
        elif key == curses.KEY_DOWN:
            board = move_down(board)
        add_new_number(board)
        if check_game_over(board):
            stdscr.addstr(10, 0, "Game Over! Press 'q' to quit.")
            stdscr.refresh()
            while stdscr.getch() != ord('q'):
                pass
            break

if __name__ == "__main__":
    curses.wrapper(game_loop)
