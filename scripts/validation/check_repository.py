"""Check source syntax, local documentation paths and SVG resources without Qt."""
import ast
from pathlib import Path
import re
import sys
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def check():
    errors = []
    sources = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT/folder).rglob('*.py')]
    sources.append(ROOT/'Meshropractor.pyw')
    for path in sources:
        try: ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except (SyntaxError, UnicodeError) as exc: errors.append(f'{path.relative_to(ROOT)}: {exc}')

    docs = [ROOT/'README.md', ROOT/'README_RU.md', ROOT/'CHANGELOG.md']
    docs += list((ROOT/'docs').rglob('*.md')) + list((ROOT/'assets').rglob('*.md'))
    links = 0
    for path in docs:
        text = path.read_text(encoding='utf-8-sig')
        # Code examples are not Markdown links.
        text = re.sub(r'```.*?```', '', text, flags=re.S)
        for target in re.findall(r'\]\(([^)]+)\)', text):
            target = target.strip().strip('<>')
            if re.match(r'[a-zA-Z][\w+.-]*:', target) or target.startswith(('#', '//')): continue
            local = unquote(target.split('#', 1)[0].split('?', 1)[0])
            if not local: continue
            links += 1
            if not (path.parent/local).exists(): errors.append(f'{path.relative_to(ROOT)}: missing link {local}')

    svgs = list((ROOT/'assets').rglob('*.svg'))
    for path in svgs:
        try:
            element = ET.parse(path).getroot()
            if not element.tag.endswith('svg'): errors.append(f'{path}: not an SVG root')
        except ET.ParseError as exc: errors.append(f'{path.relative_to(ROOT)}: {exc}')
    for name in ('logo.png', 'logo.ico', 'qr_donate.png', 'checkmark.svg', 'checkmark_dark.svg'):
        if not (ROOT/'assets'/name).is_file(): errors.append(f'Missing runtime resource: assets/{name}')

    # Detect accidental removal of a command icon rather than silently using fallbacks.
    for module, constant, group in (('texture_ribbon', 'COMMANDS', 'texture'),
                                    ('analysis_ribbon', 'COMMANDS', 'analysis')):
        tree = ast.parse((ROOT/'src'/f'{module}.py').read_text(encoding='utf-8-sig'))
        value = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == constant for t in n.targets))
        for key in ast.literal_eval(value):
            if not (ROOT/'assets/ribbon'/group/f'{key}.svg').is_file():
                errors.append(f'Missing {group} command icon: {key}')

    if errors:
        print('\n'.join(errors))
        return 1
    print(f'REPOSITORY_OK: {len(sources)} Python files, {links} local links, {len(svgs)} SVG assets')
    return 0


if __name__ == '__main__': sys.exit(check())
