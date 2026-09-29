from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

scale = 4
size = 256
canvas = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
draw = ImageDraw.Draw(canvas)

def box(values):
    return tuple(int(value * scale) for value in values)

# Soft cyan glow around a dark notification-area badge.
glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
glow_draw = ImageDraw.Draw(glow)
glow_draw.ellipse(box((20, 20, 236, 236)), fill=(34, 211, 238, 115))
glow = glow.filter(ImageFilter.GaussianBlur(9 * scale))
canvas.alpha_composite(glow)

draw.ellipse(box((20, 20, 236, 236)), fill=(8, 11, 20, 255), outline=(34, 211, 238, 255), width=8 * scale)
draw.ellipse(box((34, 34, 222, 222)), outline=(124, 58, 237, 180), width=3 * scale)

# Microphone capsule.
draw.rounded_rectangle(box((91, 48, 165, 151)), radius=35 * scale, fill=(34, 211, 238, 255))
draw.rounded_rectangle(box((106, 61, 150, 135)), radius=22 * scale, fill=(12, 23, 40, 255))
for y, width in ((74, 26), (92, 34), (110, 26)):
    x = 128 - width / 2
    draw.rounded_rectangle(box((x, y, x + width, y + 6)), radius=3 * scale, fill=(167, 243, 252, 235))

# Listening arc and stand.
draw.arc(box((70, 90, 186, 184)), start=7, end=173, fill=(167, 139, 250, 255), width=11 * scale)
draw.rounded_rectangle(box((121, 166, 135, 202)), radius=7 * scale, fill=(34, 211, 238, 255))
draw.rounded_rectangle(box((87, 196, 169, 211)), radius=7 * scale, fill=(34, 211, 238, 255))

# Tiny waveform accents that survive at 24-32 px.
for x, top, bottom in ((50, 113, 143), (61, 101, 155), (195, 101, 155), (206, 113, 143)):
    draw.rounded_rectangle(box((x, top, x + 7, bottom)), radius=3 * scale, fill=(34, 211, 238, 220))

icon = canvas.resize((size, size), Image.Resampling.LANCZOS)
png_path = ASSETS / "drew-local-voice-tray.png"
ico_path = ASSETS / "drew-local-voice-tray.ico"
icon.save(png_path)
icon.save(ico_path, format="ICO", sizes=[(16, 16), (20, 20), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print(png_path)
print(ico_path)
