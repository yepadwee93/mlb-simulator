import glob, re, os

DROPDOWN_CSS = """.nav-dropdown { position:relative; display:inline-block; }
.nav-dropdown > a { color:#ffb74d;text-decoration:none;font-weight:600; }
.nav-dropdown-menu {
  display:none; position:absolute; top:calc(100% + 6px); left:50%;
  transform:translateX(-50%);
  background:#1e2235; border:1px solid #353a52; border-radius:10px;
  min-width:160px; z-index:999; padding:6px 0; white-space:nowrap;
  box-shadow:0 8px 24px rgba(0,0,0,0.4);
}
.nav-dropdown:hover .nav-dropdown-menu,
.nav-dropdown:focus-within .nav-dropdown-menu { display:block; }
.nav-dropdown-menu a {
  display:block; padding:9px 16px; font-size:0.83rem; font-weight:600;
  text-decoration:none; color:#e8eaf0;
}
.nav-dropdown-menu a:hover { background:#252a3a; }"""

DROPDOWN_HTML = (
    '<span class="nav-dropdown">'
    '<a href="/bets">&#128176; My Bets &#9660;</a>'
    '<div class="nav-dropdown-menu">'
    '<a href="/bets">&#128176; My Bets</a>'
    '<a href="/bankroll">&#128200; Bankroll</a>'
    '<a href="/calculator">&#129518; Calculator</a>'
    '<a href="/parlay">&#127920; Parlay</a>'
    '<a href="/my-picks">&#127919; My Picks</a>'
    '</div>'
    '</span>'
)

templates = sorted(glob.glob('app/templates/*.html'))

for path in templates:
    with open(path, encoding='utf-8') as f:
        src = f.read()
    orig = src

    # 1. Inject CSS into last </style> tag
    if '.nav-dropdown' not in src:
        idx = src.rfind('</style>')
        if idx != -1:
            src = src[:idx] + '\n' + DROPDOWN_CSS + '\n' + src[idx:]

    # 2. Replace bets link (handle both old style and already-converted)
    if 'nav-dropdown-menu' not in src or 'My Bets &#9660;' not in src:
        # Still has old plain link
        src = re.sub(r'<a href="/bets"[^>]*>[^<]*My Bets[^<]*</a>', DROPDOWN_HTML, src)

    # 3. Strip standalone nav links that are now in dropdown
    for pat in [
        r'\s*<a href="/bankroll"[^>]*>[^<]{1,30}</a>',
        r'\s*<a href="/calculator"[^>]*>[^<]{1,30}</a>',
        r'\s*<a href="/parlay"[^>]*>[^<]{1,30}</a>',
        r'\s*<a href="/my-picks"[^>]*>[^<]{1,30}</a>',
    ]:
        src = re.sub(pat, '', src)

    if src != orig:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        print('Updated:', os.path.basename(path))

print('Done')
