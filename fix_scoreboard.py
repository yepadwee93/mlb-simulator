import glob, re, os

SCOREBOARD_HTML = """
<!-- ── Live Scoreboard Bar ── -->
<div id="scoreboard-bar" style="background:#12162a;border-bottom:1px solid #252836;padding:0;overflow:hidden;">
  <div id="scoreboard-inner" style="display:flex;overflow-x:auto;gap:0;scrollbar-width:none;-ms-overflow-style:none;">
    <div style="padding:7px 14px;font-size:0.7rem;color:#555870;white-space:nowrap;align-self:center;flex-shrink:0;">TODAY</div>
  </div>
</div>
<script>
(function(){
  function abbr(name) {
    var w = name.split(' ');
    return w[w.length-1].substring(0,3).toUpperCase();
  }
  function statusColor(s) {
    s = (s||'').toLowerCase();
    if (s.includes('progress')||s.includes('live')) return '#ef5350';
    if (s.includes('final')) return '#4caf50';
    return '#9aa0b8';
  }
  function renderScoreboard(games) {
    var inner = document.getElementById('scoreboard-inner');
    if (!inner) return;
    if (!games.length) { document.getElementById('scoreboard-bar').style.display='none'; return; }
    var html = '<div style="padding:7px 14px;font-size:0.7rem;color:#555870;white-space:nowrap;align-self:center;flex-shrink:0;">TODAY</div>';
    games.forEach(function(g) {
      var s = (g.status||'').toLowerCase();
      var isLive = s.includes('progress')||s.includes('live');
      var isFinal = s.includes('final');
      var timeStr = '';
      if (!isLive && !isFinal && g.game_time) {
        try { timeStr = new Date(g.game_time).toLocaleTimeString([],{hour:'numeric',minute:'2-digit'}); } catch(e){}
      } else if (isLive) {
        timeStr = (g.inning_half==='top'?'▲':'▼') + (g.inning||'');
      } else if (isFinal) {
        timeStr = 'Final';
      }
      var scoreAway = (isLive||isFinal) ? '<b style="color:#fff;">'+g.away_score+'</b>' : '';
      var scoreHome = (isLive||isFinal) ? '<b style="color:#fff;">'+g.home_score+'</b>' : '';
      var dot = '<span style="width:6px;height:6px;border-radius:50%;background:'+statusColor(g.status)+';display:inline-block;margin-right:5px;vertical-align:middle;'+(isLive?'animation:pulse 1.2s infinite;':'')+'" ></span>';
      var link = g.gamePk ? '/simulate/'+g.gamePk+'?sims=100000' : '#';
      html += '<a href="'+link+'" style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5px 14px;border-left:1px solid #1e2235;text-decoration:none;flex-shrink:0;min-width:110px;">'
        + '<div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;white-space:nowrap;">'
        + '<span style="color:#b0b8cc;">'+abbr(g.away)+' '+scoreAway+'</span>'
        + '<span style="color:#555870;font-size:0.65rem;">@</span>'
        + '<span style="color:#b0b8cc;">'+scoreHome+' '+abbr(g.home)+'</span>'
        + '</div>'
        + '<div style="font-size:0.63rem;margin-top:2px;">'+dot+'<span style="color:'+statusColor(g.status)+';">'+timeStr+'</span></div>'
        + '</a>';
    });
    inner.innerHTML = html;
  }
  fetch('/api/scores').then(function(r){return r.json();}).then(renderScoreboard).catch(function(){
    document.getElementById('scoreboard-bar').style.display='none';
  });
})();
</script>
<style>
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
#scoreboard-inner::-webkit-scrollbar { display:none; }
</style>
"""

# Templates that need the scoreboard (not index — it already has the full grid)
SKIP = {'index.html'}

templates = sorted(glob.glob('app/templates/*.html'))
changed = []

for path in templates:
    name = os.path.basename(path)
    if name in SKIP:
        continue
    with open(path, encoding='utf-8') as f:
        src = f.read()
    if 'scoreboard-bar' in src:
        continue

    # Find the closing > of the nav div (first large nav block)
    # Insert after the nav div closing tag
    # The nav is always a div ending with </div> before the page content
    # Look for the pattern: closing </div> of the top nav
    # Nav divs end with either margin-bottom:28px or similar, then </div>
    match = re.search(r'(</div>\s*\n)(\s*\n|\s*<div class="page"|\s*<div style="max-width|\s*<h1|\s*<header)', src)
    if match:
        insert_pos = match.start(1) + len(match.group(1))
        src = src[:insert_pos] + SCOREBOARD_HTML + src[insert_pos:]
        with open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        changed.append(name)

print('Added scoreboard to:', changed)
