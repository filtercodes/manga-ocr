import os
import random
import mmap
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

        # THE FIX: Use mmap for instant access to 600MB+ file without reading it into RAM
        self.corpus_path = "local_corpus.txt"
        self.file_size = 0
        self.mmap_obj = None
        
        if os.path.exists(self.corpus_path):
            self.file_size = os.path.getsize(self.corpus_path)
            self.f_obj = open(self.corpus_path, "rb")
            self.mmap_obj = mmap.mmap(self.f_obj.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self.text_corpus_fallback = ["冒険の始まりだ！魔法の剣を装備せよ。"]

    def __del__(self):
        if self.mmap_obj:
            self.mmap_obj.close()
        if hasattr(self, 'f_obj'):
            self.f_obj.close()

    def get_random_text(self, min_chars=10, max_chars=26):
        if not self.mmap_obj:
            return random.choice(self.text_corpus_fallback)

        # Pick a random byte offset and find the next valid line
        # We seek to a random spot, then read until we find a newline to align
        offset = random.randint(0, max(0, self.file_size - 1000))
        self.mmap_obj.seek(offset)
        
        # Skip the current partial line
        self.mmap_obj.readline()
        
        # Read the next full line
        line_bytes = self.mmap_obj.readline()
        if not line_bytes:
            # If we hit EOF, just pick the first line
            self.mmap_obj.seek(0)
            line_bytes = self.mmap_obj.readline()
            
        line = line_bytes.decode('utf-8', errors='ignore').strip()
        
        if len(line) < min_chars:
            # Recurse once if we got an empty/short line
            line = "宿屋で体力を回復させた。"
                
        # Slice a random chunk from the line
        if len(line) > max_chars:
            start = random.randint(0, len(line) - max_chars)
            line = line[start:start+max_chars]
        
        if random.random() < 0.2:
            keyword = random.choice(self.fantasy_keywords)
            insert_pos = random.randint(0, len(line))
            line = line[:insert_pos] + keyword + line[insert_pos:]
            line = line[:max_chars]
                
        return line

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
