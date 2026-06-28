import glob, re, os

# Add Odds History link to the accuracy link in all templates
templates = sorted(glob.glob('app/templates/*.html'))
changed = []

for path in templates:
    if 'odds_history.html' in path:
        continue
    with open(path, encoding='utf-8') as f:
        src = f.read()
    orig = src

    # Add Odds History after the Accuracy link
    src = src.replace(
        '<a href="/accuracy" style="color:#64b5f6;text-decoration:none;font-weight:600;">📊 Accuracy</a>',
        '<a href="/accuracy" style="color:#64b5f6;text-decoration:none;font-weight:600;">📊 Accuracy</a>\n    <a href="/odds-history" style="color:#ffd54f;text-decoration:none;font-weight:600;">📋 Odds</a>'
    )

    if src != orig:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        changed.append(os.path.basename(path))

print('Updated:', changed)
