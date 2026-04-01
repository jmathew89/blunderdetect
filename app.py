from flask import Flask, request, jsonify, render_template_string
import chess
import chess.engine
import chess.pgn
import chess.svg
import io
import base64
import groq
import os
import requests as http

app = Flask(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
STOCKFISH_PATH = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"
GROQ_CLIENT    = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
DEPTH          = 16

# ── Thresholds ───────────────────────────────────────────────────────────────
INACCURACY_THRESHOLD = 50
MISTAKE_THRESHOLD    = 150
BLUNDER_THRESHOLD    = 400

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

def piece_name(pt): return chess.piece_name(pt).capitalize()
def sq_name(sq):    return chess.square_name(sq)

def score_to_cp(score, cap=3000):
    if score.is_mate():
        return cap if score.mate() > 0 else -cap
    return max(-cap, min(cap, score.score()))

def classify_drop(drop):
    if drop >= BLUNDER_THRESHOLD:      return "Blunder"
    elif drop >= MISTAKE_THRESHOLD:    return "Mistake"
    elif drop >= INACCURACY_THRESHOLD: return "Inaccuracy"
    return None

def board_to_svg_b64(board, move=None, best_move_uci=None):
    """Render board as base64 SVG, highlighting the bad move and best move."""
    arrows = []
    lastmove = move

    # Show best move as a green arrow
    if best_move_uci and best_move_uci != "unknown":
        try:
            bm = chess.Move.from_uci(best_move_uci)
            arrows.append(chess.svg.Arrow(bm.from_square, bm.to_square, color="#4caf50"))
        except Exception:
            pass

    svg = chess.svg.board(
        board,
        lastmove=lastmove,
        arrows=arrows,
        size=320,
        colors={
            "square light": "#f0d9b5",
            "square dark": "#b58863",
            "square light lastmove": "#cdd16e",
            "square dark lastmove": "#aaa23a",
        }
    )
    return base64.b64encode(svg.encode()).decode()

def explain_blunder(move, label, drop, board_before, best_move_uci, score_after, best_score):
    """AI-powered explanation via Groq. Swap this function to change AI provider."""
    our_color = board_before.turn
    side = "White" if our_color == chess.WHITE else "Black"

    try:
        chat = GROQ_CLIENT.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": f"""You are a chess coach. Explain this move in 2 sentences max.

Position (FEN): {board_before.fen()}
Side to move: {side}
Move played: {move} (in chess notation: {board_before.san(move)})
Classification: {label} (eval drop: {drop} centipawns)
Best move according to engine: {best_move_uci} (in chess notation)

IMPORTANT: Only refer to moves using chess notation (like Nf3, Bxh7, O-O). Never use coordinate notation like g3g7.
Be specific and concise. Focus on the key tactical or strategic error."""
            }]
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate explanation: {e}"


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BlunderDetect</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d0d0d;
    --surface: #141414;
    --border: #2a2a2a;
    --accent: #c8a96e;
    --text: #e8e0d0;
    --text-muted: #6b6458;
    --blunder: #c94040;
    --mistake: #c97840;
    --inaccuracy: #c9b840;
    --blunder-bg: rgba(201,64,64,0.08);
    --mistake-bg: rgba(201,120,64,0.08);
    --inaccuracy-bg: rgba(201,184,64,0.08);
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    min-height: 100vh;
  }
  body::before {
    content:'';
    position:fixed; inset:0;
    background-image: repeating-conic-gradient(#1a1a1a 0% 25%, transparent 0% 50%);
    background-size: 40px 40px;
    opacity: 0.3;
    z-index:0; pointer-events:none;
  }
  .container {
    position:relative; z-index:1;
    max-width:900px; margin:0 auto;
    padding:60px 24px 80px;
  }
  header { text-align:center; margin-bottom:48px; }
  .logo {
    font-family:'Playfair Display',serif;
    font-size:clamp(36px,6vw,64px);
    font-weight:900; color:var(--accent);
    letter-spacing:-1px; line-height:1; margin-bottom:8px;
  }
  .logo span { color:var(--text); }
  .tagline { color:var(--text-muted); font-size:12px; letter-spacing:3px; text-transform:uppercase; }

  /* Tabs */
  .tabs { display:flex; gap:0; margin-bottom:16px; border:1px solid var(--border); border-radius:4px; overflow:hidden; }
  .tab {
    flex:1; padding:12px; text-align:center; cursor:pointer;
    font-size:12px; letter-spacing:2px; text-transform:uppercase;
    background:var(--surface); color:var(--text-muted);
    border:none; font-family:'IBM Plex Mono',monospace;
    transition:all 0.2s;
  }
  .tab.active { background:var(--accent); color:#0d0d0d; }
  .tab:hover:not(.active) { color:var(--text); }

  /* Upload zone */
  .upload-zone {
    border:1.5px dashed var(--border); border-radius:4px;
    padding:48px 32px; text-align:center; cursor:pointer;
    transition:all 0.2s; background:var(--surface);
    position:relative; margin-bottom:16px; display:none;
  }
  .upload-zone.visible { display:block; }
  .upload-zone:hover, .upload-zone.drag-over { border-color:var(--accent); background:rgba(200,169,110,0.04); }
  .upload-zone input[type="file"] { position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%; }
  .upload-icon { font-size:32px; margin-bottom:12px; display:block; }
  .upload-label { font-size:14px; color:var(--text-muted); line-height:1.6; }
  .upload-label strong { color:var(--accent); display:block; font-size:16px; margin-bottom:4px; }
  .file-name { margin-top:12px; font-size:13px; color:var(--accent); min-height:20px; }

  /* Paste zone */
  .paste-zone { display:none; margin-bottom:16px; }
  .paste-zone.visible { display:block; }
  textarea {
    width:100%; height:160px; background:var(--surface);
    border:1.5px solid var(--border); border-radius:4px;
    color:var(--text); font-family:'IBM Plex Mono',monospace;
    font-size:12px; padding:16px; resize:vertical;
    outline:none; transition:border-color 0.2s;
  }
  textarea:focus { border-color:var(--accent); }
  textarea::placeholder { color:var(--text-muted); }

  .analyze-btn {
    width:100%; padding:16px; background:var(--accent);
    color:#0d0d0d; border:none; border-radius:4px;
    font-family:'IBM Plex Mono',monospace; font-size:14px;
    font-weight:500; letter-spacing:2px; text-transform:uppercase;
    cursor:pointer; transition:all 0.2s; margin-bottom:40px;
  }
  .analyze-btn:hover:not(:disabled) { background:#d4b87a; transform:translateY(-1px); }
  .analyze-btn:disabled { opacity:0.4; cursor:not-allowed; }

  .loading { display:none; text-align:center; padding:40px; color:var(--text-muted); font-size:13px; letter-spacing:1px; }
  .loading.active { display:block; }
  .spinner {
    display:inline-block; width:24px; height:24px;
    border:2px solid var(--border); border-top-color:var(--accent);
    border-radius:50%; animation:spin 0.8s linear infinite; margin-bottom:12px;
  }
  @keyframes spin { to { transform:rotate(360deg); } }

  .results { display:none; }
  .results.active { display:block; }

  .game-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:20px 24px; background:var(--surface);
    border:1px solid var(--border); border-radius:4px; margin-bottom:24px;
    flex-wrap:wrap; gap:12px;
  }
  .players { font-family:'Playfair Display',serif; font-size:20px; font-weight:700; }
  .players .vs { color:var(--text-muted); font-size:14px; font-family:'IBM Plex Mono',monospace; font-weight:400; margin:0 10px; }
  .summary-pills { display:flex; gap:8px; flex-wrap:wrap; }
  .pill { padding:4px 10px; border-radius:2px; font-size:11px; letter-spacing:1px; text-transform:uppercase; font-weight:500; }
  .pill.blunder    { background:var(--blunder-bg);    color:var(--blunder);    border:1px solid rgba(201,64,64,0.3); }
  .pill.mistake    { background:var(--mistake-bg);    color:var(--mistake);    border:1px solid rgba(201,120,64,0.3); }
  .pill.inaccuracy { background:var(--inaccuracy-bg); color:var(--inaccuracy); border:1px solid rgba(201,184,64,0.3); }

  /* Move cards */
  .move-card {
    border:1px solid var(--border); border-radius:4px;
    margin-bottom:16px; overflow:hidden;
    animation:fadeUp 0.3s ease both;
  }
  @keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  .move-card.blunder    { border-left:3px solid var(--blunder);    background:var(--blunder-bg); }
  .move-card.mistake    { border-left:3px solid var(--mistake);    background:var(--mistake-bg); }
  .move-card.inaccuracy { border-left:3px solid var(--inaccuracy); background:var(--inaccuracy-bg); }

  .move-card-inner { display:flex; gap:0; }

  /* Board side */
  .board-side {
    flex-shrink:0; padding:16px;
    border-right:1px solid var(--border);
    display:flex; align-items:center; justify-content:center;
  }
  .board-side img { width:200px; height:200px; border-radius:3px; display:block; }

  /* Info side */
  .info-side { flex:1; padding:16px 20px; display:flex; flex-direction:column; gap:12px; }

  .move-header { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .move-notation { font-family:'Playfair Display',serif; font-size:20px; font-weight:700; }
  .move-label {
    font-size:11px; letter-spacing:2px; text-transform:uppercase;
    font-weight:500; padding:3px 8px; border-radius:2px;
  }
  .blunder  .move-label { color:var(--blunder);    background:rgba(201,64,64,0.15); }
  .mistake  .move-label { color:var(--mistake);    background:rgba(201,120,64,0.15); }
  .inaccuracy .move-label { color:var(--inaccuracy); background:rgba(201,184,64,0.15); }
  .eval-drop { margin-left:auto; font-size:12px; color:var(--text-muted); }

  .detail-row { display:flex; gap:10px; font-size:12px; line-height:1.6; }
  .detail-label { color:var(--text-muted); min-width:90px; flex-shrink:0; }
  .detail-value { color:var(--text); }
  .best-move { color:var(--accent); }

  .board-legend { font-size:10px; color:var(--text-muted); margin-top:6px; text-align:center; }
  .legend-bad  { color:#cdd16e; }
  .legend-good { color:#4caf50; }

  .no-issues {
    text-align:center; padding:48px; color:var(--text-muted);
    font-size:14px; background:var(--surface);
    border:1px solid var(--border); border-radius:4px;
  }
  .no-issues .big { font-size:32px; margin-bottom:8px; }

  .error-msg {
    background:rgba(201,64,64,0.1); border:1px solid rgba(201,64,64,0.3);
    color:var(--blunder); padding:16px 20px; border-radius:4px;
    font-size:13px; margin-bottom:24px; display:none;
  }
  .error-msg.active { display:block; }

  .lichess-zone { display:none; margin-bottom:16px; }
  .lichess-zone.visible { display:block; }
  .username-row { display:flex; gap:8px; margin-bottom:12px; }
  .username-input {
    flex:1; background:var(--surface); border:1.5px solid var(--border);
    border-radius:4px; color:var(--text); font-family:'IBM Plex Mono',monospace;
    font-size:14px; padding:12px 16px; outline:none; transition:border-color 0.2s;
  }
  .username-input:focus { border-color:var(--accent); }
  .username-input::placeholder { color:var(--text-muted); }
  .fetch-btn {
    padding:12px 20px; background:transparent; border:1.5px solid var(--accent);
    color:var(--accent); border-radius:4px; font-family:'IBM Plex Mono',monospace;
    font-size:13px; cursor:pointer; transition:all 0.2s; white-space:nowrap;
  }
  .fetch-btn:hover { background:var(--accent); color:#0d0d0d; }
  .fetch-btn:disabled { opacity:0.4; cursor:not-allowed; }
  .games-list { display:flex; flex-direction:column; gap:8px; }
  .game-item {
    padding:12px 16px; background:var(--surface); border:1px solid var(--border);
    border-radius:4px; cursor:pointer; transition:all 0.2s; font-size:13px;
  }
  .game-item:hover { border-color:var(--accent); }
  .game-item.selected { border-color:var(--accent); background:rgba(200,169,110,0.06); }
  .game-item-players { font-family:'Playfair Display',serif; font-size:15px; font-weight:700; margin-bottom:4px; }
  .game-item-meta { color:var(--text-muted); font-size:11px; }

  @media(max-width:600px) {
    .move-card-inner { flex-direction:column; }
    .board-side { border-right:none; border-bottom:1px solid var(--border); }
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">♟ Blunder<span>Detect</span></div>
    <div class="tagline">Chess Game Analysis · Stockfish + AI</div>
  </header>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('upload')">Upload PGN</button>
    <button class="tab" onclick="switchTab('paste')">Paste PGN</button>
    <button class="tab" onclick="switchTab('lichess')">Lichess</button>
  </div>

  <div class="upload-zone visible" id="dropZone">
    <input type="file" id="pgnFile" accept=".pgn">
    <span class="upload-icon">♙</span>
    <div class="upload-label">
      <strong>Drop your PGN file here</strong>
      or click to browse
    </div>
    <div class="file-name" id="fileName"></div>
  </div>

  <div class="paste-zone" id="pasteZone">
    <textarea id="pgnText" placeholder='Paste your PGN here...&#10;&#10;[Event "Casual Game"]&#10;[White "You"]&#10;[Black "Opponent"]&#10;&#10;1. e4 e5 2. Nf3 ...'></textarea>
  </div>

  <div class="lichess-zone" id="lichessZone">
    <div class="username-row">
      <input class="username-input" id="lichessUsername" placeholder="Lichess username (e.g. J-CUBED)" type="text">
      <button class="fetch-btn" id="fetchBtn" onclick="fetchLichessGames()">Fetch Games</button>
    </div>
    <div class="games-list" id="gamesList"></div>
  </div>

  <button class="analyze-btn" id="analyzeBtn" disabled>Analyze Game</button>

  <div class="error-msg" id="errorMsg"></div>
  <div class="loading" id="loading">
    <div class="spinner"></div>
    <div>Analyzing with Stockfish + AI — this may take a moment...</div>
  </div>

  <div class="results" id="results"></div>

  <footer>BlunderDetect · python-chess · Stockfish · Groq AI</footer>
</div>

<script>
  const dropZone   = document.getElementById('dropZone');
  const pasteZone  = document.getElementById('pasteZone');
  const lichessZone= document.getElementById('lichessZone');
  const fileInput  = document.getElementById('pgnFile');
  const fileName   = document.getElementById('fileName');
  const pgnText    = document.getElementById('pgnText');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const loading    = document.getElementById('loading');
  const results    = document.getElementById('results');
  const errorMsg   = document.getElementById('errorMsg');
  const tabs       = document.querySelectorAll('.tab');

  let mode = 'upload';
  let selectedFile = null;
  let selectedPgn  = null;
  let lichessGames = [];

  function switchTab(t) {
    mode = t;
    tabs.forEach(tab => tab.classList.toggle('active', tab.textContent.trim().toLowerCase().includes(t)));
    dropZone.classList.toggle('visible',   t === 'upload');
    pasteZone.classList.toggle('visible',  t === 'paste');
    lichessZone.classList.toggle('visible',t === 'lichess');
    selectedPgn = null;
    selectedFile = null;
    updateBtn();
  }

  function updateBtn() {
    if (mode === 'upload')       analyzeBtn.disabled = !selectedFile;
    else if (mode === 'paste')   analyzeBtn.disabled = pgnText.value.trim().length < 10;
    else if (mode === 'lichess') analyzeBtn.disabled = !selectedPgn;

  }

  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) { selectedFile = e.target.files[0]; fileName.textContent = '✓ ' + selectedFile.name; updateBtn(); }
  });
  pgnText.addEventListener('input', updateBtn);

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) { selectedFile = e.dataTransfer.files[0]; fileName.textContent = '✓ ' + selectedFile.name; updateBtn(); }
  });

  async function fetchLichessGames() {
    const username = document.getElementById('lichessUsername').value.trim();
    if (!username) return;
    const fetchBtn = document.getElementById('fetchBtn');
    const gamesList = document.getElementById('gamesList');
    fetchBtn.disabled = true;
    fetchBtn.textContent = 'Fetching...';
    gamesList.innerHTML = '<div style="color:var(--text-muted);font-size:13px;padding:12px 0;">Loading games...</div>';
    selectedPgn = null;
    updateBtn();

    try {
      const res  = await fetch(`/lichess_games?username=${encodeURIComponent(username)}`);
      const data = await res.json();
      if (data.error) { gamesList.innerHTML = `<div style="color:var(--blunder);font-size:13px;">${data.error}</div>`; return; }
      if (!data.games.length) { gamesList.innerHTML = '<div style="color:var(--text-muted);font-size:13px;">No games found.</div>'; return; }

      lichessGames = data.games;

      gamesList.innerHTML = data.games.map((g, i) => `
        <div class="game-item" onclick="selectGame(${i})" id="game-${i}">
          <div class="game-item-players">${g.white} <span style="color:var(--text-muted);font-weight:400;font-family:'IBM Plex Mono',monospace;font-size:13px;">vs</span> ${g.black}</div>
          <div class="game-item-meta">${g.opening || 'Unknown opening'} · ${g.result === 'white' ? '1-0' : g.result === 'black' ? '0-1' : '½-½'}</div>
        </div>
      `).join('');
    } catch(e) {
      gamesList.innerHTML = `<div style="color:var(--blunder);font-size:13px;">Error fetching games.</div>`;
    } finally {
      fetchBtn.disabled = false;
      fetchBtn.textContent = 'Fetch Games';
    }
  }

  function selectGame(i) {
    document.querySelectorAll('.game-item').forEach(el => el.classList.remove('selected'));
    document.getElementById(`game-${i}`).classList.add('selected');
    selectedPgn = lichessGames[i].pgn;
    updateBtn();
  }

  // Allow pressing Enter in username field
  document.getElementById('lichessUsername').addEventListener('keydown', e => {
    if (e.key === 'Enter') fetchLichessGames();
  });

  analyzeBtn.addEventListener('click', async () => {
    errorMsg.classList.remove('active');
    results.classList.remove('active');
    loading.classList.add('active');
    analyzeBtn.disabled = true;

    let body = new FormData();
    if (mode === 'upload' && selectedFile) {
      body.append('pgn', selectedFile);
    } else if (mode === 'paste') {
      body.append('pgn', new Blob([pgnText.value], { type: 'text/plain' }), 'game.pgn');
    } else if (mode === 'lichess' && selectedPgn) {
      body.append('pgn', new Blob([selectedPgn], { type: 'text/plain' }), 'game.pgn');
    }

    try {
      const res  = await fetch('/analyze', { method: 'POST', body });
      const data = await res.json();
      loading.classList.remove('active');
      if (data.error) { errorMsg.textContent = 'Error: ' + data.error; errorMsg.classList.add('active'); analyzeBtn.disabled = false; return; }
      renderResults(data);
    } catch (err) {
      loading.classList.remove('active');
      errorMsg.textContent = 'Something went wrong. Is the server running?';
      errorMsg.classList.add('active');
      analyzeBtn.disabled = false;
    }
  });

  function renderResults(data) {
    const ws = data.white_summary;
    const bs = data.black_summary;

    const scorecard = (name, color, s) => `
      <div style="flex:1;background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:16px 20px;">
        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px;">${color}</div>
        <div style="font-family:'Playfair Display',serif;font-size:18px;font-weight:700;margin-bottom:12px;">${name}</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          ${s.blunders    ? `<span class="pill blunder">${s.blunders} blunder${s.blunders>1?'s':''}</span>` : ''}
          ${s.mistakes    ? `<span class="pill mistake">${s.mistakes} mistake${s.mistakes>1?'s':''}</span>` : ''}
          ${s.inaccuracies? `<span class="pill inaccuracy">${s.inaccuracies} inaccurac${s.inaccuracies>1?'ies':'y'}</span>` : ''}
          ${!s.blunders && !s.mistakes && !s.inaccuracies ? '<span style="font-size:12px;color:var(--text-muted)">Clean game ♔</span>' : ''}
        </div>
      </div>`;

    let html = `
      <div style="display:flex;gap:12px;margin-bottom:24px;flex-wrap:wrap;">
        ${scorecard(data.white, "White", ws)}
        ${scorecard(data.black, "Black", bs)}
      </div>`;

    if (data.issues.length === 0) {
      html += `<div class="no-issues"><div class="big">♔</div>No significant issues found. Clean game!</div>`;
    } else {
      data.issues.forEach((issue, i) => {
        const cls = issue.label.toLowerCase();
        html += `
        <div class="move-card ${cls}" style="animation-delay:${i*0.05}s">
          <div class="move-card-inner">
            <div class="board-side">
              <div>
                <img src="data:image/svg+xml;base64,${issue.board_svg}" alt="Position after ${issue.move}">
                <div class="board-legend">
                  <span class="legend-bad">■</span> played &nbsp;
                  <span class="legend-good">→</span> best
                </div>
              </div>
            </div>
            <div class="info-side">
              <div class="move-header">
                <div class="move-notation">${issue.move}</div>
                <span class="move-label">${issue.label}</span>
                <span class="eval-drop">${issue.drop > 0 ? '−' : '+'}${Math.abs(issue.drop)} cp</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Player</span>
                <span class="detail-value" style="color:var(--accent)">${issue.player} (${issue.color})</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Best move</span>
                <span class="detail-value best-move">${issue.best_move}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Explanation</span>
                <span class="detail-value">${issue.explanation}</span>
              </div>
            </div>
          </div>
        </div>`;
      });
    }

    results.innerHTML = html;
    results.classList.add('active');
    analyzeBtn.disabled = false;
  }
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "pgn" not in request.files:
        return jsonify({"error": "No PGN file uploaded"}), 400

    pgn_text = request.files["pgn"].read().decode("utf-8")
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return jsonify({"error": "Could not parse PGN file"}), 400

    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        return jsonify({"error": f"Could not start Stockfish: {e}"}), 500

    board       = game.board()
    move_number = 1
    issues      = []

    try:
        for node in game.mainline():
            move         = node.move
            board_before = board.copy()

            mover_color   = board.turn
            info_before   = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            cp_before     = score_to_cp(info_before["score"].white())
            best_move_uci = str(info_before["pv"][0]) if info_before.get("pv") else "unknown"
            best_score    = info_before["score"].pov(board.turn)

            # Convert best move to SAN before pushing
            try:
                best_move_san = board.san(chess.Move.from_uci(best_move_uci))
            except Exception:
                best_move_san = best_move_uci

            board.push(move)

            # Never flag a move that delivered checkmate — it's always correct
            if board.is_checkmate():
                move_number += 1
                continue

            info_after  = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            score_after = info_after["score"].pov(not board.turn)
            cp_after    = score_to_cp(info_after["score"].white())

            # mover_color is who just moved. White wants higher scores, Black wants lower.
            if mover_color == chess.WHITE:
                drop = cp_before - cp_after   # positive = got worse for white
            else:
                drop = cp_after - cp_before   # positive = got worse for black (score went up = bad for black)
            label = classify_drop(drop)

            full_move = (move_number + 1) // 2
            side      = "..." if board.turn == chess.WHITE else "."
            move_str  = f"{full_move}{side} {node.san()}"
            player    = white if mover_color == chess.WHITE else black

            if label:
                explanation = explain_blunder(move, label, drop, board_before, best_move_san, score_after, best_score)
                board_svg   = board_to_svg_b64(board, move=move, best_move_uci=best_move_uci)
                issues.append({
                    "move":        move_str,
                    "label":       label,
                    "drop":        drop,
                    "best_move":   best_move_san,
                    "explanation": explanation,
                    "board_svg":   board_svg,
                    "player":      player,
                    "color":       "white" if mover_color == chess.WHITE else "black",
                })

            move_number += 1
    finally:
        engine.quit()

    return jsonify({
        "white": white, "black": black,
        "issues": issues,
        "white_summary": {
            "blunders":     sum(1 for i in issues if i["color"]=="white" and i["label"]=="Blunder"),
            "mistakes":     sum(1 for i in issues if i["color"]=="white" and i["label"]=="Mistake"),
            "inaccuracies": sum(1 for i in issues if i["color"]=="white" and i["label"]=="Inaccuracy"),
        },
        "black_summary": {
            "blunders":     sum(1 for i in issues if i["color"]=="black" and i["label"]=="Blunder"),
            "mistakes":     sum(1 for i in issues if i["color"]=="black" and i["label"]=="Mistake"),
            "inaccuracies": sum(1 for i in issues if i["color"]=="black" and i["label"]=="Inaccuracy"),
        },
    })


@app.route("/lichess_games", methods=["GET"])
def lichess_games():
    username = request.args.get("username", "").strip()
    if not username:
        return jsonify({"error": "No username provided"}), 400

    try:
        # Fetch last 10 games as ndjson
        res = http.get(
            f"https://lichess.org/api/games/user/{username}",
            params={"max": 10, "clocks": "false", "evals": "false", "opening": "true"},
            headers={"Accept": "application/x-ndjson"},
            timeout=15,
        )
        if res.status_code == 404:
            return jsonify({"error": f"User '{username}' not found on Lichess"}), 404
        if res.status_code != 200:
            return jsonify({"error": f"Lichess API error: {res.status_code}"}), 500

        import json
        games = []
        for line in res.text.strip().split("\n"):
            if not line.strip(): continue
            g = json.loads(line)

            white  = g.get("players", {}).get("white", {}).get("user", {}).get("name", "?")
            black  = g.get("players", {}).get("black", {}).get("user", {}).get("name", "?")
            result = g.get("winner", "draw")
            opening = g.get("opening", {}).get("name", "")
            game_id = g.get("id", "")

            # Fetch PGN separately for each game
            pgn_res = http.get(
                f"https://lichess.org/game/export/{game_id}",
                params={"clocks": "false", "evals": "false"},
                headers={"Accept": "application/x-chess-pgn"},
                timeout=10,
            )
            pgn = pgn_res.text if pgn_res.status_code == 200 else ""

            games.append({
                "id":      game_id,
                "white":   white,
                "black":   black,
                "result":  result,
                "opening": opening,
                "pgn":     pgn,
            })

        return jsonify({"games": games})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n  ♟  BlunderDetect is running!")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000)
