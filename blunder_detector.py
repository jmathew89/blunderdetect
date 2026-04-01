import chess
import chess.engine
import chess.pgn
import sys

# ── Thresholds (in centipawns) ──────────────────────────────────────────────
INACCURACY_THRESHOLD = 50
MISTAKE_THRESHOLD    = 100
BLUNDER_THRESHOLD    = 300

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

def piece_name(pt): return chess.piece_name(pt).capitalize()
def sq_name(sq):    return chess.square_name(sq)

# ── Material count for a board ───────────────────────────────────────────────
def material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        PIECE_VALUES[pt] * len(board.pieces(pt, color))
        for pt in PIECE_VALUES
    )

# ── Detect if a square is a fork (piece attacks 2+ valuable targets) ─────────
def find_fork_targets(board: chess.Board, sq: chess.Square, attacker_color: chess.Color):
    """Return list of valuable pieces the piece on sq attacks."""
    piece = board.piece_at(sq)
    if not piece:
        return []
    targets = []
    for target_sq in board.attacks(sq):
        target = board.piece_at(target_sq)
        if target and target.color != attacker_color and target.piece_type != chess.PAWN:
            targets.append((target_sq, target))
    return targets

# ── Detect pins ──────────────────────────────────────────────────────────────
def is_pinned(board: chess.Board, sq: chess.Square) -> bool:
    piece = board.piece_at(sq)
    if not piece:
        return False
    return board.is_pinned(piece.color, sq)

# ── Smart explanation engine ─────────────────────────────────────────────────
def explain_blunder(move: chess.Move, label: str, drop: int,
                    board_before: chess.Board,
                    best_move_uci: str,
                    score_before: chess.engine.Score,
                    score_after: chess.engine.Score,
                    best_score: chess.engine.Score) -> str:

    our_color  = board_before.turn
    opp_color  = not our_color
    board_after = board_before.copy()
    board_after.push(move)

    reasons = []

    # ── 1. Missed checkmate ──────────────────────────────────────────────────
    if best_score.is_mate() and best_score.mate() > 0:
        mate_in = best_score.mate()
        move_label = "move" if mate_in == 1 else f"{mate_in} moves"
        reasons.append(f"you missed checkmate in {move_label}! The best move was {best_move_uci}")

    # ── 2. Allowed checkmate ─────────────────────────────────────────────────
    elif score_after.is_mate() and score_after.mate() < 0:
        mate_in = abs(score_after.mate())
        move_label = "next move" if mate_in == 1 else f"{mate_in} moves"
        reasons.append(f"it allows your opponent to checkmate you in {move_label}")

    else:
        # ── 3. Hung the moved piece ──────────────────────────────────────────
        to_sq = move.to_square
        moved_piece = board_after.piece_at(to_sq)
        if moved_piece:
            attacked = board_after.is_attacked_by(opp_color, to_sq)
            defended = board_after.is_attacked_by(our_color, to_sq)
            if attacked and not defended:
                reasons.append(
                    f"your {piece_name(moved_piece.piece_type)} moved to {sq_name(to_sq)} "
                    f"where it is undefended and can be captured for free"
                )
            elif attacked and defended:
                # Check if trade is losing (attacker is lower value)
                attackers = board_after.attackers(opp_color, to_sq)
                if attackers:
                    min_attacker = min(
                        PIECE_VALUES[board_after.piece_at(a).piece_type]
                        for a in attackers if board_after.piece_at(a)
                    )
                    our_val = PIECE_VALUES[moved_piece.piece_type]
                    if min_attacker < our_val:
                        reasons.append(
                            f"your {piece_name(moved_piece.piece_type)} moved to {sq_name(to_sq)} "
                            f"where it can be captured by a lower-value piece — a losing trade"
                        )

        # ── 4. Left a piece hanging ──────────────────────────────────────────
        for sq in chess.SQUARES:
            piece = board_after.piece_at(sq)
            if piece and piece.color == our_color and sq != to_sq:
                attacked  = board_after.is_attacked_by(opp_color, sq)
                defended  = board_after.is_attacked_by(our_color, sq)
                if attacked and not defended and piece.piece_type != chess.PAWN:
                    reasons.append(
                        f"it left your {piece_name(piece.piece_type)} on {sq_name(sq)} "
                        f"hanging — undefended and under attack"
                    )
                    break  # report the most important one

        # ── 5. Allowed a fork ────────────────────────────────────────────────
        if not reasons:
            for sq in chess.SQUARES:
                p = board_after.piece_at(sq)
                if p and p.color == opp_color:
                    targets = find_fork_targets(board_after, sq, opp_color)
                    if len(targets) >= 2:
                        target_names = " and ".join(
                            f"{piece_name(t.piece_type)} on {sq_name(tsq)}"
                            for tsq, t in targets[:2]
                        )
                        reasons.append(
                            f"it allowed the opponent's {piece_name(p.piece_type)} on "
                            f"{sq_name(sq)} to fork your {target_names}"
                        )
                        break

        # ── 6. Missed a winning capture ──────────────────────────────────────
        if not reasons:
            try:
                best_move = chess.Move.from_uci(best_move_uci)
                if board_before.is_capture(best_move):
                    captured = board_before.piece_at(best_move.to_square)
                    capturing = board_before.piece_at(best_move.from_square)
                    if captured and capturing:
                        reasons.append(
                            f"the best move {best_move_uci} would have captured your opponent's "
                            f"{piece_name(captured.piece_type)} with your "
                            f"{piece_name(capturing.piece_type)} — a free material gain you missed"
                        )
            except Exception:
                pass

        # ── 7. Moved into check exposure ────────────────────────────────────
        if not reasons:
            our_king = board_after.king(our_color)
            if our_king and board_after.is_attacked_by(opp_color, our_king):
                reasons.append("it exposed your king to attack")

        # ── 8. Positional fallback with context ─────────────────────────────
        if not reasons:
            mat_before = material(board_before, our_color) - material(board_before, opp_color)
            mat_after  = material(board_after,  our_color) - material(board_after,  opp_color)
            mat_loss   = mat_before - mat_after
            if mat_loss >= 250:
                lost_val = mat_loss
                # Find what piece value was lost
                if   lost_val >= 800: piece_lost = "queen"
                elif lost_val >= 450: piece_lost = "rook"
                elif lost_val >= 250: piece_lost = "minor piece (bishop or knight)"
                else:                 piece_lost = "material"
                reasons.append(f"it resulted in losing a {piece_lost} without compensation")
            elif drop >= 300:
                reasons.append(
                    f"it gave the opponent a decisive positional advantage "
                    f"(the engine preferred {best_move_uci} instead)"
                )
            elif drop >= 100:
                reasons.append(
                    f"it was a positional mistake — {best_move_uci} was significantly stronger"
                )
            else:
                reasons.append(
                    f"it was slightly inaccurate; {best_move_uci} keeps a better position"
                )

    # ── Build final sentence ─────────────────────────────────────────────────
    explanation = f"{label}: {reasons[0]}"
    if len(reasons) > 1:
        explanation += f". Also: {reasons[1]}"
    explanation += "."
    return explanation


# ── Label a move based on eval drop ─────────────────────────────────────────
def classify_drop(drop: int) -> str | None:
    if drop >= BLUNDER_THRESHOLD:   return "Blunder"
    elif drop >= MISTAKE_THRESHOLD: return "Mistake"
    elif drop >= INACCURACY_THRESHOLD: return "Inaccuracy"
    return None

# ── Clamp engine score ───────────────────────────────────────────────────────
def score_to_cp(score: chess.engine.Score, cap: int = 3000) -> int:
    if score.is_mate():
        return cap if score.mate() > 0 else -cap
    return max(-cap, min(cap, score.score()))

# ── Main analysis ────────────────────────────────────────────────────────────
def analyze_game(pgn_path: str, stockfish_path: str, depth: int = 18):
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)

    if game is None:
        print("Error: no game found in PGN file.")
        return

    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")
    print(f"\n{'─'*50}")
    print(f"  {white} vs {black}")
    print(f"{'─'*50}\n")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board  = game.board()
    move_number = 1
    issues_found = 0

    for node in game.mainline():
        move         = node.move
        board_before = board.copy()

        info_before  = engine.analyse(board, chess.engine.Limit(depth=depth))
        score_before = info_before["score"].pov(board.turn)
        cp_before    = score_to_cp(info_before["score"].white())
        best_move_uci = str(info_before["pv"][0]) if info_before.get("pv") else "unknown"
        best_score   = info_before["score"].pov(board.turn)

        board.push(move)

        info_after   = engine.analyse(board, chess.engine.Limit(depth=depth))
        score_after  = info_after["score"].pov(not board.turn)  # from mover's perspective
        cp_after     = score_to_cp(info_after["score"].white())

        if board.turn == chess.WHITE:
            drop = -(cp_after - cp_before)
        else:
            drop = cp_before - cp_after

        label = classify_drop(drop)

        full_move = (move_number + 1) // 2
        side      = "..." if board.turn == chess.WHITE else "."
        move_str  = f"{full_move}{side} {node.san()}"

        if label:
            issues_found += 1
            print(f"Move {move_str:<18}  {label:<12}  (eval drop: {drop:+} cp)")
            print(f"  └─ Best move was:  {best_move_uci}")
            explanation = explain_blunder(
                move, label, drop, board_before,
                best_move_uci, score_before, score_after, best_score
            )
            print(f"  └─ Explanation:    {explanation}")
            print()

        move_number += 1

    engine.quit()
    print(f"\n{'─'*50}")
    print(f"  Total issues flagged: {issues_found}")
    print(f"{'─'*50}\n")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python blunder_detector.py <game.pgn> <path/to/stockfish>")
        sys.exit(1)
    analyze_game(pgn_path=sys.argv[1], stockfish_path=sys.argv[2])
