import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from manga_ocr_dev.synthetic_data_generator.dirt_pipeline import build_dirt_pipeline
from pathlib import Path

class RetroGenerator:
    def __init__(self, map_crt_path="manga-ocr/assets/maps/zelda_lttp.png", map_gb_path="manga-ocr/assets/maps/zelda_gb.png"):
        self.map_crt = Image.open(map_crt_path).convert("RGB")
        self.map_gb = Image.open(map_gb_path).convert("RGB")
        
        # Define the font pool with their respective native pixel sizes
        self.font_pool = [
            {"path": "manga-ocr/assets/fonts/PixelMplus12-Regular.ttf", "size": 12},
            {"path": "manga-ocr/assets/fonts/Nosutaru-dotMPlusH-10-Regular.ttf", "size": 10},
            {"path": "manga-ocr/assets/fonts/misaki_gothic.ttf", "size": 12}, # Misaki is 8 but often used at 12
            {"path": "manga-ocr/assets/fonts/JF-Dot-Ayu20.ttf", "size": 20},
            {"path": "manga-ocr/assets/fonts/Mona12.ttf", "size": 12},
            {"path": "manga-ocr/assets/fonts/Mona12-Bold.ttf", "size": 12},
        ]
        
        # Pre-build the 3 pipelines
        self.pipelines = {
            "crt": build_dirt_pipeline("crt"),
            "lcd": build_dirt_pipeline("lcd"),
            "xbrz": build_dirt_pipeline("xbrz")
        }
        
        self.fantasy_keywords = ["剣", "魔法", "装備", "魔王", "勇者", "冒険", "宿屋", "薬草", "洞窟", "城"]

        # THE FIX: Load entire corpus into RAM for O(1) access
        corpus_path = "local_corpus.txt"
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                self.text_corpus = f.read().splitlines()
        else:
            self.text_corpus = ["冒険の始まりだ！魔法の剣を装備せよ。"]

    def get_random_text(self, min_chars=10, max_chars=26):
        # THE FIX: Instant RAM selection
        line = random.choice(self.text_corpus)
        
        if len(line) < max_chars:
            # Simple fallback if line is too short
            return (line + " " + "冒険の書を記録します。")[:max_chars]
                
        start = random.randint(0, len(line) - max_chars)
        length = random.randint(min_chars, max_chars)
        chunk = line[start:start+length]
        
        if random.random() < 0.2:
            keyword = random.choice(self.fantasy_keywords)
            insert_pos = random.randint(0, len(chunk))
            chunk = chunk[:insert_pos] + keyword + chunk[insert_pos:]
            chunk = chunk[:max_chars]
                
            return chunk
        return chunk

    def get_background_crop(self, map_img, width=700, height=200):
        map_w, map_h = map_img.size
        x = random.randint(0, map_w - width)
        y = random.randint(0, map_h - height)
        return map_img.crop((x, y, x + width, y + height))

    def draw_ui_box(self, draw, width, height, style=None):
        if style is None:
            style = random.choice(["solid", "translucent_blue", "wood"])
            
        margin = 30
        box_coords = [margin, margin, width - margin, height - margin]
        
        if style == "solid":
            draw.rectangle(box_coords, fill=(0, 0, 0, 255))
            draw.rectangle(box_coords, outline=(255, 255, 255, 255), width=2)
        elif style == "translucent_blue":
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(box_coords, fill=(0, 0, 150, 180))
            overlay_draw.rectangle(box_coords, outline=(255, 255, 255, 255), width=2)
            return overlay
        elif style == "wood":
            draw.rectangle(box_coords, fill=(60, 30, 10, 255))
            draw.rectangle(box_coords, outline=(150, 100, 50, 255), width=3)
        return None

    def render_pixel_text(self, text, font_path, font_size, scale=4):
        font = ImageFont.truetype(font_path, font_size)
        dummy_img = Image.new('RGBA', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        txt_img = Image.new('RGBA', (tw + 6, th + 10), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.fontmode = "1"
        
        color = random.choice([(255, 255, 255), (224, 224, 224), (255, 255, 0)])
        txt_draw.text((3, 3), text, font=font, fill=(0, 0, 0, 255))
        txt_draw.text((2, 2), text, font=font, fill=color)
        
        new_size = (txt_img.width * scale, txt_img.height * scale)
        return txt_img.resize(new_size, resample=Image.NEAREST)

    def generate_one(self):
        modes = ["crt", "lcd", "xbrz"]
        weights = [0.6, 0.2, 0.2]
        mode = random.choices(modes, weights=weights)[0]
        
        target_map = self.map_gb if mode == "lcd" else self.map_crt
        bg = self.get_background_crop(target_map)
        width, height = bg.size
        
        raw_text = self.get_random_text()
        
        # Select random font
        font_config = random.choice(self.font_pool)
        font_size = font_config["size"]
        
        # Calculate scaling to make final letter height ~40-60px
        # We need integer scales for pixel art to stay crisp with NEAREST scaling
        scale = 4 if font_size <= 12 else 3
        
        # Max dimensions inside the UI box (700x200 canvas with 30px margins = 640x140 safe area)
        safe_width = width - 80   # adding a bit of extra padding
        safe_height = height - 80
        
        # Native safe dimensions before scaling
        native_safe_width = safe_width // scale
        native_safe_height = safe_height // scale
        
        # Approximate chars per line based on font_size (assuming square characters)
        chars_per_line = max(1, native_safe_width // font_size)
        
        # Max lines we can fit (giving a few extra native pixels for line spacing)
        line_height = font_size + 4 
        max_lines = max(1, native_safe_height // line_height)
        
        # Truncate raw_text so it absolutely fits inside the safe area
        max_chars = chars_per_line * max_lines
        raw_text = raw_text[:max_chars]
        
        # Wrap text
        text = "\n".join([raw_text[i:i+chars_per_line] for i in range(0, len(raw_text), chars_per_line)])
        
        canvas = bg.convert("RGBA")
        draw = ImageDraw.Draw(canvas)
        overlay = self.draw_ui_box(draw, width, height)
        if overlay:
            canvas = Image.alpha_composite(canvas, overlay)
            
        text_layer = self.render_pixel_text(text, font_config["path"], font_config["size"], scale=scale)
        px = (width - text_layer.width) // 2
        py = (height - text_layer.height) // 2
        
        final_ui = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        final_ui.paste(text_layer, (px, py), text_layer)
        canvas = Image.alpha_composite(canvas, final_ui)
        
        img_np = np.array(canvas.convert("RGB"))
        active_pipeline = self.pipelines[mode]
        dirty_img = active_pipeline(image=img_np)["image"]
        
        return dirty_img, raw_text, font_config["path"]

def run_test():
    import sys
    import os
    gen = RetroGenerator()
    img, text, _ = gen.generate_one()
    base_path = "manga-ocr/assets/examples/retro_test"
    ext = ".jpg"
    out_path = f"{base_path}{ext}"
    counter = 1
    while os.path.exists(out_path):
        out_path = f"{base_path}_{counter}{ext}"
        counter += 1
    Image.fromarray(img).save(out_path)
    print(f"Generated test image with text: {text}")
    print(f"Saved to {out_path}")
    sys.exit(0)

if __name__ == "__main__":
    run_test()
