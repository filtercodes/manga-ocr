import os
import random
import mmap
import json
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageStat
from manga_ocr_dev.synthetic_data_generator.dirt_pipeline import build_dirt_pipeline
from pathlib import Path

class RetroGenerator:
    def __init__(self, 
                 map_crt_path="manga-ocr/assets/maps/zelda_lttp.png", 
                 map_gb_path="manga-ocr/assets/maps/zelda_gb.png",
                 pixel_art_dir="pixel_art",
                 corpus_path="structured_corpus.json"):
        # Load the base Zelda maps for the 'classic' 60% category
        self.map_crt = Image.open(map_crt_path).convert("RGB")
        self.map_gb = Image.open(map_gb_path).convert("RGB")
        
        # Load stress-test background paths (Lazy loading to save RAM/Time)
        self.pixel_art_backgrounds = []
        if os.path.exists(pixel_art_dir):
            for f in os.listdir(pixel_art_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif')):
                    self.pixel_art_backgrounds.append(os.path.join(pixel_art_dir, f))

        # Define the font pool with explicit safety routing for complex Kanji
        self.font_pool = [
            {"path": "manga-ocr/assets/fonts/PixelMplus12-Regular.ttf", "size": 12, "kanji_safe": True},
            {"path": "manga-ocr/assets/fonts/Nosutaru-dotMPlusH-10-Regular.ttf", "size": 10, "kanji_safe": False},
            {"path": "manga-ocr/assets/fonts/misaki_gothic.ttf", "size": 12, "kanji_safe": False}, 
            {"path": "manga-ocr/assets/fonts/JF-Dot-Ayu20.ttf", "size": 20, "kanji_safe": True},
            {"path": "manga-ocr/assets/fonts/Mona12.ttf", "size": 12, "kanji_safe": True},
            {"path": "manga-ocr/assets/fonts/Mona12-Bold.ttf", "size": 12, "kanji_safe": False},
        ]
        
        # Pre-build the 3 pipelines (CRT emulation, LCD grid, xBRZ smoothing)
        self.pipelines = {
            "crt": build_dirt_pipeline("crt"),
            "lcd": build_dirt_pipeline("lcd"),
            "xbrz": build_dirt_pipeline("xbrz")
        }
        
        # NEW CORPUS ENGINE: Load the 225k lines from the structured JSON
        self.corpus = []
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                for game in raw_data:
                    game_name = game.get("game", "Unknown")
                    for block in game.get("blocks", []):
                        for line in block:
                            self.corpus.append({"game": game_name, "text": line})
        
        if not self.corpus:
            self.corpus = [{"game": "Fallback", "text": "冒険の始まりだ！魔法の剣を装備せよ。"}]

        # Regex to detect common speaker patterns like 【Name】
        self.speaker_regex = re.compile(r"^【(.*?)】(.*)$", re.DOTALL)
        # NEW: Regex to detect complex Kanji blocks (CJK Unified Ideographs)
        self.kanji_regex = re.compile(r'[一-龯]')
        # pool of characters for the 10% Name Overfitting Protection
        self.nonsense_chars = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"

        self.mmap_obj = None
        self.file_size = 0

    def get_random_sample(self):
        """
        Picks a random line from our structured corpus.
        Splits it into (Speaker, Dialogue) if brackets are detected.
        Includes 10% 'Nonsense Name' protection to prevent overfitting.
        """
        # DEBUG TRAIT: Set to True to ONLY generate images with character names
        FORCE_SPEAKER = True
        
        while True:
            sample = random.choice(self.corpus)
            raw_text = sample["text"]
            match = self.speaker_regex.match(raw_text)
            
            if match:
                speaker = match.group(1)
                dialogue = match.group(2).strip()
                # THE OVERFITTING GUARD: 10% chance to swap name for random Katakana
                if random.random() < 0.10:
                    speaker = "".join(random.choices(self.nonsense_chars, k=len(speaker)))
                return speaker, dialogue
                
            if not FORCE_SPEAKER:
                return None, raw_text

    def wrap_text(self, text, font_size):
        """
        Wraps text to fit 1 to 4 lines, utilizing longer corpus strings.
        Returns the wrapped visual text and the exact string for ground truth.
        """
        target_lines = random.randint(1, 4)
        base_chars = 14 if font_size <= 12 else 8
        chars_per_line = random.randint(base_chars - 2, base_chars + 4)
        max_chars = chars_per_line * target_lines
        rendered_text = text[:max_chars]
        wrapped_text = "\n".join([rendered_text[i:i+chars_per_line] for i in range(0, len(rendered_text), chars_per_line)])
        return wrapped_text, rendered_text

    def get_background_crop(self, mode="classic", width=700, height=200):
        """
        Selects a background crop based on the 80/20 background split.
        """
        if mode == "pristine":
            return Image.new("RGB", (width, height), (0, 0, 0))

        # 80/20 Background Split Engine
        if self.pixel_art_backgrounds and random.random() < 0.80:
            bg_path = random.choice(self.pixel_art_backgrounds)
            try:
                bg_img = Image.open(bg_path)
                if bg_img.mode == 'P' and 'transparency' in bg_img.info:
                    bg_img = bg_img.convert('RGBA')
                bg_img = bg_img.convert("RGB")
            except:
                bg_img = random.choice([self.map_crt, self.map_gb])
        else:
            bg_img = random.choice([self.map_crt, self.map_gb])

        bw, bh = bg_img.size
        target_w = min(width, bw)
        target_h = min(height, bh)
        x = random.randint(0, bw - target_w)
        y = random.randint(0, bh - target_h)
        return bg_img.crop((x, y, x + target_w, y + target_h))

    def draw_ui_box(self, canvas, border_coords, fill_coords, style=None, gap=None, gap_top=None, separator_y=None, sep_w=2, sep_a=255):
        """
        Renders UI boxes with a decoupled background fill that authentically overflows the borders.
        """
        draw = ImageDraw.Draw(canvas)
        bx0, by0, bx1, by1 = border_coords
        
        if style == "solid":
            # Draw the massive, unified background block
            draw.rectangle(fill_coords, fill=(0, 0, 0, 255))
            if separator_y:
                draw.line([bx0, separator_y, bx1, separator_y], fill=(255, 255, 255, sep_a), width=sep_w)
            
            # Draw the inset border lines
            if gap:
                g_start, g_end = gap
                draw.line([bx0, by0, g_start, by0], fill=(255, 255, 255, 255), width=2)
                draw.line([g_end, by0, bx1, by0], fill=(255, 255, 255, 255), width=2)
                draw.line([bx0, by0, bx0, by1], fill=(255, 255, 255, 255), width=2)
                draw.line([bx1, by0, bx1, by1], fill=(255, 255, 255, 255), width=2)
                draw.line([bx0, by1, bx1, by1], fill=(255, 255, 255, 255), width=2)
            else:
                draw.rectangle(border_coords, outline=(255, 255, 255, 255), width=2)
                
        elif style == "translucent_blue":
            overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(fill_coords, fill=(0, 0, 150, 180))
            if separator_y:
                overlay_draw.line([bx0, separator_y, bx1, separator_y], fill=(255, 255, 255, sep_a), width=sep_w)
            if gap:
                g_start, g_end = gap
                overlay_draw.line([bx0, by0, g_start, by0], fill=(255, 255, 255, 255), width=2)
                overlay_draw.line([g_end, by0, bx1, by0], fill=(255, 255, 255, 255), width=2)
                overlay_draw.line([bx0, by0, bx0, by1], fill=(255, 255, 255, 255), width=2)
                overlay_draw.line([bx1, by0, bx1, by1], fill=(255, 255, 255, 255), width=2)
                overlay_draw.line([bx0, by1, bx1, by1], fill=(255, 255, 255, 255), width=2)
            else:
                overlay_draw.rectangle(border_coords, outline=(255, 255, 255, 255), width=2)
            canvas.alpha_composite(overlay)
            
        elif style == "wood":
            draw.rectangle(fill_coords, fill=(60, 30, 10, 255))
            if separator_y:
                draw.line([bx0, separator_y, bx1, separator_y], fill=(150, 100, 50, sep_a), width=sep_w)
            b_color = (150, 100, 50, 255)
            if gap:
                g_start, g_end = gap
                draw.line([bx0, by0, g_start, by0], fill=b_color, width=3)
                draw.line([g_end, by0, bx1, by0], fill=b_color, width=3)
                draw.line([bx0, by0, bx0, by1], fill=b_color, width=3)
                draw.line([bx1, by0, bx1, by1], fill=b_color, width=3)
                draw.line([bx0, by1, bx1, by1], fill=b_color, width=3)
            else:
                draw.rectangle(border_coords, outline=b_color, width=3)
        return False

    def render_pixel_text(self, text, font_path, font_size, color=(255, 255, 255), shadow=False):
        """
        Renders PURE, BINARY pixel text. Supports 1px hard drop shadows.
        Automatically corrects negative font offsets to prevent clipping.
        """
        font = ImageFont.truetype(font_path, font_size)
        line_spacing = max(4, int(font_size * 0.4)) 
        
        dummy_img = Image.new('RGBA', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        # Pillow returns (left, top, right, bottom)
        bbox = dummy_draw.multiline_textbbox((0, 0), text, font=font, spacing=line_spacing)
        left, top, right, bottom = bbox
        
        # Exact width and height of the text block
        tw = right - left
        th = bottom - top
        
        # Add padding to guarantee we don't hit the edge during OpenCV dilation later
        pad = 6
        canvas_w = tw + (pad * 2)
        canvas_h = th + (pad * 2)
        
        # Shift the drawing coordinates by the negative offset to keep text in bounds
        draw_x = -left + pad
        draw_y = -top + pad
        
        # 1. FOREGROUND LAYER
        fg_img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
        fg_draw = ImageDraw.Draw(fg_img)
        fill_color = tuple(list(color[:3]) + [255])
        fg_draw.multiline_text((draw_x, draw_y), text, font=font, fill=fill_color, spacing=line_spacing)
        
        # --- THE SWISS CHEESE FIX: Ultra-Sensitive Alpha Guillotine ---
        fg_arr = np.array(fg_img)
        # Drop threshold to 10 to catch ultra-faint TrueType vector strokes
        alpha_mask = fg_arr[:, :, 3] > 10 
        
        # Force RGB to target color, delete AA halos
        fg_arr[:, :, 0] = np.where(alpha_mask, color[0], 0)
        fg_arr[:, :, 1] = np.where(alpha_mask, color[1], 0)
        fg_arr[:, :, 2] = np.where(alpha_mask, color[2], 0)
        fg_arr[:, :, 3] = np.where(alpha_mask, 255, 0)
        fg_img = Image.fromarray(fg_arr, 'RGBA')
        
        if shadow:
            # 2. SHADOW LAYER
            sh_img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
            sh_draw = ImageDraw.Draw(sh_img)
            # Offset shadow by exactly 1 native pixel
            sh_draw.multiline_text((draw_x + 1, draw_y + 1), text, font=font, fill=(0, 0, 0, 255), spacing=line_spacing)
            
            sh_arr = np.array(sh_img)
            s_mask = sh_arr[:, :, 3] > 10
            
            # Force shadow to pure black
            sh_arr[:, :, 0] = 0
            sh_arr[:, :, 1] = 0
            sh_arr[:, :, 2] = 0
            sh_arr[:, :, 3] = np.where(s_mask, 255, 0)
            sh_img = Image.fromarray(sh_arr, 'RGBA')
            
            # 3. COMPOSITE
            sh_img.alpha_composite(fg_img)
            return sh_img, line_spacing
        else:
            return fg_img, line_spacing

    def generate_one(self):
        """
        The Orchestrator. Implements 60/30/10 split and four spatial archetypes.
        """
        # --- 1. THE SPLIT & SIZING ---
        roll = random.random()
        category = "classic" if roll < 0.60 else ("stress" if roll < 0.90 else "pristine")

        font_config = random.choice(self.font_pool)
        font_size = font_config["size"]
        scale = 4 if font_size <= 12 else 3 
        
        # --- Contextual Singleton ---
        # 8-bit fonts cannot render complex Kanji without becoming unreadable blobs.
        # We loop until we find a sample that fits the font's physical resolution.
        while True:
            speaker, dialogue = self.get_random_sample()
            full_test_string = dialogue if not speaker else f"{speaker}{dialogue}"
            
            # Count exact number of Kanji in the string
            kanji_count = len(self.kanji_regex.findall(full_test_string))
            
            # Use explicit kanji_safe flag from font config
            if not font_config.get("kanji_safe", True):
                # 8-bit bitmap fonts (Misaki, Nosutaru, Bold)
                # Allow exactly 0 or 1 Kanji to provide context without illegible blobs.
                if kanji_count <= 1:
                    break 
            else:
                # Vector or large fonts can handle infinite Kanji
                break 

        archetype = random.choice([1, 2, 3]) if speaker else 0
        
        # --- Embedded Border is impossible without a UI Box.
        if category == "stress" and archetype == 2:
            archetype = random.choice([1, 3]) # Reroll to Naked Top-Line or Inline Bracket
            
        visual_dialogue, gt_dialogue = self.wrap_text(dialogue, font_size)

        # --- 2. PRE-RENDER BACKGROUND & LUMINANCE SCOUTER (Regional Detection) ---
        # Calculate padding first so we can crop the background immediately
        pad_top, pad_bottom = random.randint(10, 200), random.randint(10, 200)
        pad_left, pad_right = random.randint(10, 250), random.randint(10, 250)
        
        # Estimate canvas size to grab the crop before rendering
        # Account for speaker name length if present
        max_chars = len(visual_dialogue.split('\n')[0])
        if speaker and len(speaker) > max_chars:
            max_chars = len(speaker)
            
        est_w = (max_chars * font_size * scale) + pad_left + pad_right + 100 # added 100px safety
        est_h = (len(visual_dialogue.split('\n')) * font_size * scale * 2) + pad_top + pad_bottom + 100
        bg = self.get_background_crop(mode=category, width=est_w, height=est_h)
        
        # Calculate the average color and brightness of the target region where text starts
        # We crop a rough 150x50 patch to scout the local background noise
        crop_box = (pad_left, pad_top, min(bg.width, pad_left + 150), min(bg.height, pad_top + 50))
        stat = ImageStat.Stat(bg.crop(crop_box))
        bg_mean = stat.mean[:3] # Average [R, G, B]
        bg_stddev = stat.stddev[:3] # NEW: Extract the Standard Deviation (Noise/Texture)
        
        # Standard luminance formula to determine if background is light or dark
        luminance = 0.299 * bg_mean[0] + 0.587 * bg_mean[1] + 0.114 * bg_mean[2]
        
        # NEW: Calculate overall Texture/Noise Chaos
        noise_level = 0.299 * bg_stddev[0] + 0.587 * bg_stddev[1] + 0.114 * bg_stddev[2]

        # --- 3. THE COLOR PALETTE & ARCHETYPE RULES (Guaranteed Contrast) ---
        outline_color = None
        use_shadow = False
        
        if category == "classic":
            # Classic UI boxes are usually dark, so text is always bright
            fg_color = random.choice([(255, 255, 255), (255, 255, 0), (0, 255, 255), (224, 224, 224)])
        elif category == "pristine":
            # Inverted Dark-on-Light for pure topology regularization
            fg_color = (20, 20, 20)
        else: # STRESS CATEGORY (Adversarial)
            # 50% chance for DBZ Style, OR 100% chance if the background is highly chaotic/noisy.
            # If noise_level > 40, the background has harsh contrasting spikes. We MUST deploy an outline shield.
            if random.random() < 0.5 or noise_level > 40:
                # Force the text color to perfectly match the background's average color
                shift = random.randint(-15, 15)
                fg_color = tuple(int(max(0, min(255, c + shift))) for c in bg_mean)
                
                # The outline acts as a physical wall against the background noise
                outline_color = (0, 0, 0, 255) if luminance > 100 else (255, 255, 255, 255)
                use_shadow = False
            
            # 50% chance: Dark Half Style (Contrasting Text + 1px Drop Shadow)
            else:
                # This is now only allowed to execute if noise_level <= 40 (smooth backgrounds)
                if luminance > 128:
                    # Bright background: Use crisp Black text, NO drop shadow
                    fg_color = (0, 0, 0)
                    use_shadow = False
                else:
                    # Dark background: Use crisp White text, WITH Black drop shadow
                    fg_color = (255, 255, 255)
                    use_shadow = True

        # Ground Truth standardization
        if archetype in [1, 2, 3]:
            text_gt = f"【{speaker}】{gt_dialogue}"
        else:
            text_gt = gt_dialogue
        visual_text = visual_dialogue

        # Determine variant
        arch2_style = random.choice(["embedded", "separator"]) if archetype == 2 else None

        # --- 5. RENDER TEXT LAYER(S) ---
        if archetype == 1:
            # Naked Name: Independent layers for jittering
            name_layer_tiny, f_spacing = self.render_pixel_text(speaker, font_config["path"], font_size, color=fg_color, shadow=use_shadow)
            dialogue_layer_tiny, _ = self.render_pixel_text(visual_dialogue, font_config["path"], font_size, color=fg_color, shadow=use_shadow)
            
            name_scaled = name_layer_tiny.resize((name_layer_tiny.width * scale, name_layer_tiny.height * scale), Image.NEAREST)
            dialogue_scaled = dialogue_layer_tiny.resize((dialogue_layer_tiny.width * scale, dialogue_layer_tiny.height * scale), Image.NEAREST)
            
            # --- 50/50 Standard vs. Jitter Split ---
            logical_line_height = (font_size + f_spacing) * scale
            
            if random.random() < 0.50:
                # 50% Standard Layout (Usopp/Sanji Baseline)
                # Name sits flush left, directly above the dialogue with natural line spacing.
                name_offset_x = 0
                name_offset_y = 0
                base_dialogue_y = name_scaled.height
            else:
                # 50% Spatial Jitter (Topological Resilience)
                # Force the name to physically break away from the dialogue block
                name_offset_x = random.randint(-40, -10)
                name_offset_y = random.randint(-30, -10)
                # Add an extra half-line gap to clearly disconnect them visually
                base_dialogue_y = name_scaled.height + int(logical_line_height * 0.5) 
            
            shift_x = abs(name_offset_x)
            shift_y = abs(name_offset_y)
            
            total_w = max(name_scaled.width, dialogue_scaled.width) + shift_x
            total_h = base_dialogue_y + dialogue_scaled.height + shift_y
            
            text_layer_scaled = Image.new('RGBA', (total_w, total_h), (0, 0, 0, 0))
            text_layer_scaled.alpha_composite(name_scaled, (shift_x + name_offset_x, shift_y + name_offset_y))
            text_layer_scaled.alpha_composite(dialogue_scaled, (shift_x, shift_y + base_dialogue_y))
            
        elif archetype == 2: # Embedded Border
            if arch2_style == "separator":
                # Continuous fractional scaling (5% to 50% larger)
                name_scale_factor = random.uniform(1.05, 1.50)
                name_fs = int(font_size * name_scale_factor)
            else:
                name_fs = font_size

            text_layer_tiny, _ = self.render_pixel_text(visual_dialogue, font_config["path"], font_size, color=fg_color, shadow=use_shadow)
            text_layer_scaled = text_layer_tiny.resize((text_layer_tiny.width * scale, text_layer_tiny.height * scale), Image.NEAREST)
            
            name_layer_tiny, _ = self.render_pixel_text(speaker, font_config["path"], name_fs, color=fg_color, shadow=use_shadow)
            name_layer_scaled = name_layer_tiny.resize((name_layer_tiny.width * scale, name_layer_tiny.height * scale), Image.NEAREST)
        
        else: # Standard / Inline (Archetype 0 and 3)
            if archetype == 3:
                # Sub-divide Archetype 3 into historical game variants
                style_roll = random.random()
                
                if style_roll < 0.33:
                    # 3A: Classic Inline (e.g., Seiken Densetsu) -> Name「Dialogue
                    delim = random.choice(["「", "『", "："])
                    visual_full = f"{speaker}{delim}{visual_text}"
                    
                elif style_roll < 0.66:
                    # 3B: Newline Quote (e.g., Onimusha) -> Name \n「Dialogue
                    delim = random.choice(["「", "『"])
                    visual_full = f"{speaker}\n{delim}{visual_text}"
                    
                else:
                    # 3C: Explicit Source Brackets (e.g., Hunter x Hunter) -> 【Name】\nDialogue
                    # We use native brackets in the visual image itself
                    bracket_left, bracket_right = random.choice([("【", "】"), ("[", "]")])  #, ("<", ">")])
                    visual_full = f"{bracket_left}{speaker}{bracket_right}\n{visual_text}"
                    
                # Render the combined text
                text_layer_tiny, _ = self.render_pixel_text(visual_full, font_config["path"], font_size, color=fg_color, shadow=use_shadow)
                text_layer_scaled = text_layer_tiny.resize((text_layer_tiny.width * scale, text_layer_tiny.height * scale), Image.NEAREST)
                
            else:
                # Archetype 0: Pure Dialogue (No Speaker)
                text_layer_tiny, _ = self.render_pixel_text(visual_text, font_config["path"], font_size, color=fg_color, shadow=use_shadow)
                text_layer_scaled = text_layer_tiny.resize((text_layer_tiny.width * scale, text_layer_tiny.height * scale), Image.NEAREST)

        # Initialize default telemetry variables for the outline
        outline_iters = 0
        outline_shape_name = "None"

        # --- 6. THE CONTOUR (Dynamic Post-Scale Dilation) ---
        if outline_color:
            np_text = np.array(text_layer_scaled)
            alpha_mask = np_text[:, :, 3] 
            
            # Base kernel size guarantees 1 iteration = 1 native pixel thickness
            k_size = (scale * 2) + 1
            
            # Randomize Geometry (80% Cross for authentic retro, 20% Rect for fat/blocky)
            if random.random() < 0.80:
                kernel_shape = cv2.MORPH_CROSS
                outline_shape_name = "cross"
            else:
                kernel_shape = cv2.MORPH_RECT
                outline_shape_name = "rect"
                
            kernel = cv2.getStructuringElement(kernel_shape, (k_size, k_size))
            
            # Randomize Thickness (Iterations) safely based on Font Size
            if font_size <= 12:
                outline_iters = 1 # Small fonts must stay thin to survive
            else:
                outline_iters = random.choice([1, 2]) # Large fonts can handle fatter outlines
                
            dilated_mask = cv2.dilate(alpha_mask, kernel, iterations=outline_iters)
            
            # Construct the outline layer
            outline_np = np.zeros_like(np_text)
            outline_np[:, :, :3] = outline_color[:3]
            outline_np[:, :, 3] = dilated_mask
            
            outline_layer = Image.fromarray(outline_np, 'RGBA')
            final_text_layer = Image.new('RGBA', text_layer_scaled.size, (0, 0, 0, 0))
            final_text_layer.alpha_composite(outline_layer)
            final_text_layer.alpha_composite(text_layer_scaled)
            
            # Apply identical dynamic parameters to the floating name layer (Archetype 2)
            if archetype == 2:
                np_name = np.array(name_layer_scaled)
                n_mask = cv2.dilate(np_name[:, :, 3], kernel, iterations=outline_iters)
                n_outline = np.zeros_like(np_name)
                n_outline[:, :, :3] = outline_color[:3]
                n_outline[:, :, 3] = n_mask
                final_name_layer = Image.new('RGBA', name_layer_scaled.size, (0, 0, 0, 0))
                final_name_layer.alpha_composite(Image.fromarray(n_outline, 'RGBA'))
                final_name_layer.alpha_composite(name_layer_scaled)
                name_layer_scaled = final_name_layer
        else:
            final_text_layer = text_layer_scaled

        # --- 7. CONSTRUCTION (The Canvas Awareness) ---
        # We must ensure the pad_top is large enough to prevent the floating name (Archetype 2)
        # from being pushed into negative Y-space.
        # Define the width of the inner content
        content_w = final_text_layer.width

        if archetype == 2:
            min_required_top = name_layer_scaled.height + 40 # 40px buffer for border and padding
            if pad_top < min_required_top:
                pad_top = min_required_top
                
            # If the name is wider than the dialogue, expand the true content width
            required_name_w = name_layer_scaled.width + 30
            if content_w < required_name_w:
                content_w = required_name_w

        name_y = pad_top
        dialogue_y = pad_top
        separator_y = None
        name_x = pad_left + 15 # Indent slightly to breathe from the left wall
        border_margin_top = 15

        if archetype == 2:
            if arch2_style == "embedded":
                border_margin_top = (name_layer_scaled.height // 2) + 10
                name_y = (pad_top - border_margin_top) - (name_layer_scaled.height // 2)

            # Calculate the vertical layout shifts before sizing the canvas
            if arch2_style == "separator":
                border_margin_top = 25
                line_gap = random.randint(10, 20)
                separator_y = name_y + name_layer_scaled.height + line_gap
                dialogue_y = separator_y + line_gap

        canvas_w = content_w + pad_left + pad_right
        # Canvas height stretches to the bottom of the pushed-down dialogue
        canvas_h = dialogue_y + final_text_layer.height + pad_bottom

        if bg.size != (canvas_w, canvas_h):
            bg = bg.resize((canvas_w, canvas_h), Image.LANCZOS)
        canvas = bg.convert("RGBA")

        # --- 8. DRAW UI BOX & COMPOSITE ---
        border_margin_sides, border_margin_bottom = 15, 15

        # Dynamic line thickness (1-3px) and opacity (50-255)
        sep_w, sep_a = random.randint(1, 3), random.randint(50, 255)

        # 1. Calculate where the white border line sits
        border_coords = [
            pad_left - border_margin_sides,
            pad_top - border_margin_top,
            pad_left + content_w + border_margin_sides,
            canvas_h - pad_bottom + border_margin_bottom
        ]
        
        # 2. Calculate the massive background fill (Universally overflows the border by 10px)
        fill_inset = 10
        fill_coords = [
            border_coords[0] - fill_inset,
            border_coords[1] - fill_inset,
            border_coords[2] + fill_inset,
            border_coords[3] + fill_inset
        ]

        if archetype == 2 and arch2_style == "embedded":
            # We push it up to name_y minus a 15px safety buffer.
            fill_coords[1] = min(fill_coords[1], name_y - 15)
        
        if category == "classic":
            box_style = random.choice(["solid", "translucent_blue", "wood"])
            
            if archetype == 2 and arch2_style == "embedded":
                gap_start, gap_end = name_x - 5, name_x + name_layer_scaled.width + 5
                self.draw_ui_box(canvas, border_coords, fill_coords, style=box_style, gap=(gap_start, gap_end), gap_top=name_y)
            elif archetype == 2 and arch2_style == "separator":
                # Feed the dynamic variables to the UI renderer
                self.draw_ui_box(canvas, border_coords, fill_coords, style=box_style, separator_y=separator_y, sep_w=sep_w, sep_a=sep_a)
            else:
                self.draw_ui_box(canvas, border_coords, fill_coords, style=box_style)
                
        # --- 9. FINAL COMPOSITE ---
        if archetype == 2:
            canvas.alpha_composite(name_layer_scaled, (name_x, name_y))
        canvas.alpha_composite(final_text_layer, (pad_left, dialogue_y))
        
        # --- 10. DEGRADATION & PIPELINE ---
        img_np = np.array(canvas.convert("RGB"))
        if category != "pristine":
            # Quarantine xBRZ. 
            # xBRZ destroys delicate camouflaged outlines, so we ban it from the "stress" category.
            available_pipelines = ["crt", "lcd"]
            
            # We only allow xBRZ on classic UI boxes where high contrast makes it safe (and bulky).
            if category == "classic":
                available_pipelines.append("xbrz")
                
            mode_filter = random.choice(available_pipelines)
            active_pipeline = self.pipelines[mode_filter]
            dirty_img = active_pipeline(image=img_np)["image"]
        else:
            dirty_img = img_np
            mode_filter = "None"
            
        # Construct the telemetry dictionary for observability
        debug_dict = {
            "category": category,
            "archetype": archetype,
            "bg_luminance": round(luminance, 2),
            "bg_noise": round(noise_level, 2),
            "fg_color": str(fg_color),
            "outline_color": str(outline_color) if outline_color else "None",
            "outline_iters": outline_iters,           # NEW: Thickness 
            "outline_shape": outline_shape_name,      # NEW: Geometry
            "has_shadow": use_shadow,
            "pipeline_used": mode_filter
        }

        return dirty_img, text_gt, font_config["path"], debug_dict

def run_test():
    import sys
    import os
    gen = RetroGenerator()
    img, text, _, debug = gen.generate_one()
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
