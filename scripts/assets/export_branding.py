"""Package the selected PNG logo into a padded app PNG and multi-resolution ICO."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = Image.open(ROOT / 'docs/design/startup/meshropractor-mark-contoured.png').convert('RGBA')
    # Remove empty export margins, keeping the original concept untouched.
    bounds = source.getchannel('A').point(lambda value: 255 if value > 32 else 0).getbbox()
    if bounds is None: raise ValueError('Logo is empty')
    mark = source.crop(bounds)
    mark.thumbnail((432, 432), Image.Resampling.LANCZOS)
    icon = Image.new('RGBA', (512, 512))
    icon.alpha_composite(mark, ((512-mark.width)//2, (512-mark.height)//2))
    icon.save(ROOT / 'assets/logo.png')
    icon.save(ROOT / 'assets/logo.ico', sizes=[(s,s) for s in (16,20,24,32,40,48,64,128,256)])


if __name__ == '__main__': main()
