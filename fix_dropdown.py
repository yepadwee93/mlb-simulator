import glob, re, os

FULL_DROPDOWN = (
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

for path in sorted(glob.glob('app/templates/*.html')):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    orig = src
    # Replace any nav-dropdown span (regardless of content) with full version
    src = re.sub(
        r'<span class="nav-dropdown">.*?</span>',
        FULL_DROPDOWN,
        src,
        flags=re.DOTALL
    )
    if src != orig:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        print('Fixed:', os.path.basename(path))

print('Done')
