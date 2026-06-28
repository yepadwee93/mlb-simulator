import glob, os

OLD_CSS = """.nav-dropdown { position:relative; display:inline-block; }
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

NEW_CSS = """.nav-dropdown { position:relative; display:inline-block; }
.nav-dropdown > a { color:#ffb74d;text-decoration:none;font-weight:600; }
.nav-dropdown-menu {
  visibility:hidden; opacity:0; pointer-events:none;
  position:absolute; top:100%; left:50%;
  transform:translateX(-50%);
  background:#1e2235; border:1px solid #353a52; border-radius:10px;
  min-width:160px; z-index:999; padding:12px 0 6px 0; white-space:nowrap;
  box-shadow:0 8px 24px rgba(0,0,0,0.4);
  transition:opacity 0.15s, visibility 0.15s;
  margin-top:0;
}
.nav-dropdown:hover .nav-dropdown-menu,
.nav-dropdown:focus-within .nav-dropdown-menu {
  visibility:visible; opacity:1; pointer-events:auto;
}
.nav-dropdown-menu a {
  display:block; padding:9px 16px; font-size:0.83rem; font-weight:600;
  text-decoration:none; color:#e8eaf0;
}
.nav-dropdown-menu a:hover { background:#252a3a; }"""

changed = []
for path in sorted(glob.glob('app/templates/*.html')):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    if OLD_CSS in src:
        src = src.replace(OLD_CSS, NEW_CSS, 1)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        changed.append(os.path.basename(path))

print('Updated:', changed)
