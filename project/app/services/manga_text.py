import argparse
import os
import json

from PIL import Image, ImageDraw, ImageFont, ImageFilter

def add_onomatopoeia_with_shadow(
    input_path: str,
    output_path: str,
    text: str,
    position: tuple[int, int],
    font_path: str,
    font_size: int,
    rotate_deg: float,
):
    base = Image.open(input_path).convert("RGBA")
    font = ImageFont.truetype(font_path, font_size)

    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=6)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    pad = 50
    layer = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    shadow_layer = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_draw.text(
        (pad + 8, pad + 8),
        text,
        font=font,
        fill=(0, 0, 0, 180),
        stroke_width=8,
        stroke_fill=(0, 0, 0, 200),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(2))

    draw.text(
        (pad, pad),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=6,
        stroke_fill=(0, 0, 0, 255),
    )

    merged = Image.alpha_composite(shadow_layer, layer)
    rotated = merged.rotate(rotate_deg, expand=True, resample=Image.Resampling.BICUBIC)

    base.alpha_composite(rotated, dest=position)
    base.save(output_path)

def resolve_position(placement_hint: str, scene_match: str):
    if placement_hint == "center":
        return (180, 220)
    elif placement_hint == "top":
        return (180, 80)
    elif placement_hint == "bottom":
        return (180, 420)
    elif placement_hint == "left":
        return (60, 220)
    elif placement_hint == "right":
        return (320, 220)

    if scene_match == "urban_street":
        return (160, 140)
    return (180, 220)

def resolve_font_size(intensity: str, style_hint: str):
    base = 110
    if intensity == "weak":
        base = 80
    elif intensity == "medium":
        base = 110
    elif intensity == "strong":
        base = 140

    if style_hint == "bold":
        base += 10
    return base

def resolve_rotate_deg(scene_match: str, style_hint: str):
    if scene_match == "urban_street":
        return -12
    if style_hint == "bold":
        return -8
    return -5

def run(audio_id: str):
    base_dir = f"/home/team-009/project/data/results/{audio_id}"

    input_path = os.path.join(base_dir, "09_manga_image.png")
    output_path = os.path.join(base_dir, "10_manga_text.png")
    onomato_path = os.path.join(base_dir, "08_onomatopoeia.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input image not found: {input_path}")

    if not os.path.exists(onomato_path):
        raise FileNotFoundError(f"onomatopoeia json not found: {onomato_path}")

    with open(onomato_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = data.get("primary_onomatopoeia", "ザワザワ")
    intensity = data.get("intensity", "medium")
    style_hint = data.get("style_hint", "bold")
    placement_hint = data.get("placement_hint", "center")
    scene_match = data.get("scene_match", "")

    position = resolve_position(placement_hint, scene_match)
    font_size = resolve_font_size(intensity, style_hint)
    rotate_deg = resolve_rotate_deg(scene_match, style_hint)

    font_path = "/home/team-009/project/fonts/NotoSansJP-Bold.ttf"

    add_onomatopoeia_with_shadow(
        input_path=input_path,
        output_path=output_path,
        text=text,
        position=position,
        font_path=font_path,
        font_size=font_size,
        rotate_deg=rotate_deg,
    )

    result = {
        "audio_id": audio_id,
        "input": input_path,
        "output": output_path,
        "text_source": "08_onomatopoeia.json",
        "onomatopoeia": text,
        "intensity": intensity,
        "style_hint": style_hint,
        "placement_hint": placement_hint,
        "position": position,
        "font_size": font_size,
        "rotate_deg": rotate_deg,
    }

    meta_path = os.path.join(base_dir, "10_manga_text.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-id", required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    result = run(args.audio_id)

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()