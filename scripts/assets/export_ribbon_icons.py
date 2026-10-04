"""One-time export of the original programmatic SVGs to editable assets."""
import importlib
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / 'src'))


def export(module, function, group, pairs):
    target = importlib.import_module(module)
    original_pixmap, original_icon = target.QPixmap, target.QIcon
    captured = []
    class Pixmap:
        def loadFromData(self, data, *args): captured.append(bytes(data)); return True
    class Icon:
        def __init__(self, *args): pass
        def addPixmap(self, *args): pass
    target.QPixmap, target.QIcon = Pixmap, Icon
    try:
        for key, args in pairs:
            captured.clear(); getattr(target, function)(*args)
            if not captured: continue
            data = next((data for data in captured if b'width="48"' in data), captured[0])
            folder = ROOT / 'assets/ribbon' / group; folder.mkdir(parents=True, exist_ok=True)
            path = folder / (str(key) + '.svg')
            # Never overwrite a manually edited asset.
            if not path.exists(): path.write_bytes(data)
    finally: target.QPixmap, target.QIcon = original_pixmap, original_icon


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    import repair_ribbon, placement_ribbon, display_ribbon, texture_ribbon, settings_icons
    for module, function, group, keys in (
        ('repair_ribbon', 'repair_icon', 'repair', repair_ribbon.REPAIR_COMMANDS),
        ('placement_ribbon', 'placement_icon', 'placement', placement_ribbon.PLACEMENT_COMMANDS),
        ('display_ribbon', 'display_icon', 'display', display_ribbon.DISPLAY_COMMANDS),
        ('texture_ribbon', 'texture_icon', 'texture', texture_ribbon.COMMANDS),
        ('settings_icons', 'settings_icon', 'settings', settings_icons.SETTINGS_COMMANDS),
        ('workspace_icons', 'workspace_icon', 'workspace', ['platform','part','triangle','plane','smooth','component','brush','rectangle','clear','support','selected','tree','manual','preview']),
        ('tool_ribbon', 'main_icon', 'main', ['Новый проект','Загрузить проект','Сохранить проект','Сохранить проект как','Импорт детали','Сохранить выбранные детали как','Сохранить все в папку','Выгрузить деталь']),
        ('tool_ribbon', 'tool_icon', 'tools', range(7)),
        ('model_tool_ribbon', 'model_tool_icon', 'tools', __import__('model_tool_ribbon').COMMANDS.keys() - {'union','difference','intersection'}),
    ): export(module, function, group, [(key, (key,)) for key in keys])
    import ast
    source=ast.parse((ROOT/'src/cad_dialog.py').read_text(encoding='utf-8-sig'))
    function=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='cad_icon')
    svg=next(ast.literal_eval(node.value) for node in function.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='svg')
    path=ROOT/'assets/ribbon/tools/cad.svg'
    if not path.exists():path.write_bytes(svg)
    for key,index in {'move':3,'rotate':4,'scale':5,'mirror':6}.items():
        path=ROOT/'assets/ribbon/placement'/f'{key}.svg'
        if not path.exists():path.write_bytes((ROOT/'assets/ribbon/tools'/f'{index}.svg').read_bytes())
    from analysis_ribbon import COMMANDS
    motifs={
        'view':'M7 33 24 24 41 33 24 42Z', 'intersections':'M7 9H28V30H7ZM20 20H41V41H20Z',
        'trapping':'M8 8H20V39H8ZM32 8H44V39H32ZM20 24H32M26 18 32 24 26 30',
        'walls':'M9 8H16V40H9ZM32 8H39V40H32ZM16 24H32M20 20 16 24 20 28M28 20 32 24 28 28',
        'cavities':'M6 6H42V42H6ZM17 17H31V31H17Z', 'risks':'M24 5 44 41H4ZM24 16V29M24 35V36',
        'slices':'M6 39V7M6 39H42M12 32 18 21 25 25 32 12 41 17',
        'time':'M24 8V24L35 30', 'cost':'M5 10H43V38H5ZM14 24H34M24 15V33',
        'material':'M24 6 40 14V34L24 42 8 34V14ZM8 14 24 24 40 14M24 24V42',
        'volume':'M8 12 24 5 40 12V34L24 42 8 34ZM8 12 24 22 40 12M24 22V42',
        'density':'M5 6H43V42H5ZM10 12H19V22H10ZM27 12H37V22H27ZM10 29H19V37H10ZM27 29H37V37H27Z',
        'dimensions':'M8 12H40V36H8ZM4 6H44M4 3V9M44 3V9M4 42H44',
        'mass_center':'M24 4V44M4 24H44', 'bounds':'M6 13 24 5 42 13V35L24 43 6 35ZM6 13 24 23 42 13M24 23V43',
        'distance':'M6 35 38 9M7 24 16 36M28 7 40 20', 'thickness':'M9 8V40M39 8V40M9 24H39M15 18 9 24 15 30M33 18 39 24 33 30',
        'actual':'M7 6H33V42H7ZM14 16H26M14 25H26M14 34H24M26 34 33 40 44 25',
        'precision':'M4 24H44M10 18V30M17 21V27M24 18V30M31 21V27M38 18V30',
        'report':'M10 5H30L40 15V43H10ZM30 5V15H40M16 22H34M16 30H34M16 37H29',
        'template':'M6 6H42V42H6ZM6 16H42M17 16V42M22 23H36M22 32H36',
    }
    folder=ROOT/'assets/ribbon/analysis';folder.mkdir(exist_ok=True)
    for key in COMMANDS:
        path=folder/(key+'.svg')
        extra='<circle cx="24" cy="24" r="19" fill="none" stroke="#acc6d4" stroke-width="2"/>' if key in ('time','mass_center') else ''
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">{extra}<path d="{motifs[key]}" fill="none" stroke="#79bfd1" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        if not path.exists():path.write_text(svg,encoding='utf-8')
    print('SVG assets exported')
