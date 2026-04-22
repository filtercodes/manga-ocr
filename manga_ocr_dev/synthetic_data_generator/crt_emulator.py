import cv2
import numpy as np
import albumentations as A

class CRTDistortion(A.DualTransform):
    """
    Simulates physical CRT monitor artifacts including geometric barrel distortion,
    luminance-dependent scanlines, RGB phosphor masks, chromatic aberration, and multi-pass bloom.
    """
    def __init__(
        self,
        k1_range=(-0.05, 0.05),
        k2_range=(-0.02, 0.02),
        bloom_scale_range=(0.0, 0.3),
        scanline_alpha_range=(0.2, 0.6),
        mask_types=("aperture", "slot", "shadow", "none"),
        always_apply=False,
        p=0.5
    ):
        super().__init__(p=p)
        self.k1_range = k1_range
        self.k2_range = k2_range
        self.bloom_scale_range = bloom_scale_range
        self.scanline_alpha_range = scanline_alpha_range
        self.mask_types = mask_types

    def get_params_dependent_on_data(self, params, data):
        return {
            "k1": self.py_random.uniform(*self.k1_range),
            "k2": self.py_random.uniform(*self.k2_range),
            "bloom_scale": self.py_random.uniform(*self.bloom_scale_range),
            "scanline_alpha": self.py_random.uniform(*self.scanline_alpha_range),
            "mask_type": self.py_random.choice(self.mask_types),
            "converge_x": self.py_random.uniform(-0.4, 0.4),
            "converge_y": self.py_random.uniform(-0.4, 0.4)
        }

    def apply(self, img, k1=0.0, k2=0.0, bloom_scale=0.0, scanline_alpha=0.0, mask_type="none", converge_x=0.0, converge_y=0.0, **params):
        # 1. Chromatic Aberration (electron gun misalignment)
        img = self._apply_chromatic_aberration(img, converge_x, converge_y)
        
        # 2. Multi-pass Bloom (Halation)
        img = self._apply_bloom(img, bloom_scale)
        
        # 3. Scanlines (Dynamic Width based on luminance)
        img = self._apply_scanlines(img, scanline_alpha)
        
        # 4. Phosphor Mask (Procedural layout)
        if mask_type != "none":
            img = self._apply_phosphor_mask(img, mask_type)
        
        # 5. Geometric Curvature (Barrel/Pincushion distortion)
        img = self._apply_barrel_distortion(img, k1, k2)
        
        return img

    def apply_to_bboxes(self, bboxes, k1=0.0, k2=0.0, **params):
        if not bboxes or (k1 == 0 and k2 == 0):
            return bboxes
        
        warped_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:4]
            points = []
            for x in np.linspace(x_min, x_max, 5):
                for y in np.linspace(y_min, y_max, 5):
                    nx = x * 2.0 - 1.0
                    ny = y * 2.0 - 1.0
                    r2 = nx**2 + ny**2
                    distortion = 1.0 + k1 * r2 + k2 * (r2**2)
                    nx_dist = nx * distortion
                    ny_dist = ny * distortion
                    x_dist = (nx_dist + 1.0) / 2.0
                    y_dist = (ny_dist + 1.0) / 2.0
                    points.append((x_dist, y_dist))
            
            points = np.array(points)
            new_xmin = np.clip(points[:, 0].min(), 0.0, 1.0)
            new_ymin = np.clip(points[:, 1].min(), 0.0, 1.0)
            new_xmax = np.clip(points[:, 0].max(), 0.0, 1.0)
            new_ymax = np.clip(points[:, 1].max(), 0.0, 1.0)
            
            new_bbox = [new_xmin, new_ymin, new_xmax, new_ymax] + list(bbox[4:])
            warped_bboxes.append(new_bbox)
        return warped_bboxes

    def get_transform_init_args_names(self):
        return ("k1_range", "k2_range", "bloom_scale_range", "scanline_alpha_range", "mask_types")

    def _apply_chromatic_aberration(self, img, cx, cy):
        if len(img.shape) < 3 or img.shape[2] != 3:
            # If grayscale, convert to BGR first to apply color shifts
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        b, g, r = cv2.split(img)
        rows, cols = img.shape[:2]
        
        M_r = np.float32([[1, 0, cx], [0, 1, cy]])
        M_b = np.float32([[1, 0, -cx], [0, 1, -cy]])
        
        r_shifted = cv2.warpAffine(r, M_r, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        b_shifted = cv2.warpAffine(b, M_b, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        
        return cv2.merge((b_shifted, g, r_shifted))

    def _apply_bloom(self, img, bloom_scale):
        if bloom_scale <= 0:
            return img
        
        img_float = img.astype(np.float32)
        accum = np.zeros_like(img_float)
        
        # MAME style progressive downscale/blur blending
        weights = [1.0, 0.64, 0.32, 0.16, 0.08, 0.06, 0.04, 0.02, 0.01]
        
        for level, w in enumerate(weights):
            ksize = (2**level) * 2 + 1
            blurred = cv2.GaussianBlur(img_float, (ksize, ksize), 0)
            accum += blurred * w
            
        accum = accum * bloom_scale
        blended = img_float + accum
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _apply_scanlines(self, img, scanline_alpha):
        h, w = img.shape[:2]
        img_float = img.astype(np.float32) / 255.0
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Luminance calculation (BGR)
            L = 0.114 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.299 * img_float[:, :, 2]
        else:
            L = img_float
            img_float = np.expand_dims(img_float, axis=-1)

        spacing = 3.0
        alpha = 0.5
        beta = 0.5
        # Line width expands based on luminance
        W = alpha + beta * L
        
        y_coords = np.arange(h).reshape(-1, 1)
        distance = np.mod(y_coords, spacing)
        distance = np.where(distance > spacing / 2, spacing - distance, distance)
        
        I = np.exp(- (distance**2) / (W**2 + 1e-5))
        if len(img.shape) == 3:
            I = np.expand_dims(I, axis=-1)
            
        blended = img_float * (1.0 - scanline_alpha + scanline_alpha * I)
        return np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    def _apply_phosphor_mask(self, img, mask_type):
        h, w = img.shape[:2]
        if len(img.shape) < 3 or img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        # THE FIX: Inactive phosphors only darken by 15%, leaving text highly legible
        floor = 0.85 
        
        if mask_type == "aperture":
            kernel = np.array([[[1.0, floor, floor], [floor, 1.0, floor], [floor, floor, 1.0]]], dtype=np.float32)
            mask = np.tile(kernel, (h, int(np.ceil(w / 3)), 1))[:h, :w, :]
            
        elif mask_type == "slot":
            kernel = np.full((6, 6, 3), floor, dtype=np.float32)
            kernel[0:3, 0:2, 0] = 1.0 
            kernel[0:3, 2:4, 1] = 1.0 
            kernel[0:3, 4:6, 2] = 1.0 
            kernel[3:6, 3:5, 0] = 1.0 
            kernel[3:6, 5:6, 1] = 1.0 
            kernel[3:6, 0:1, 1] = 1.0 
            kernel[3:6, 1:3, 2] = 1.0 
            mask = np.tile(kernel, (int(np.ceil(h / 6)), int(np.ceil(w / 6)), 1))[:h, :w, :]
            
        elif mask_type == "shadow":
            kernel = np.full((4, 6, 3), floor, dtype=np.float32)
            kernel[0:2, 0:2, 0] = 1.0 
            kernel[0:2, 2:4, 1] = 1.0 
            kernel[0:2, 4:6, 2] = 1.0 
            kernel[2:4, 3:5, 0] = 1.0 
            kernel[2:4, 5:6, 1] = 1.0 
            kernel[2:4, 0:1, 1] = 1.0 
            kernel[2:4, 1:3, 2] = 1.0 
            mask = np.tile(kernel, (int(np.ceil(h / 4)), int(np.ceil(w / 6)), 1))[:h, :w, :]
            
        else:
            mask = np.ones((h, w, 3), dtype=np.float32)
            
        img_float = img.astype(np.float32)
        # We don't need a massive brightboost anymore because we aren't deleting 66% of the light
        return np.clip(img_float * mask, 0, 255).astype(np.uint8)

    def _apply_barrel_distortion(self, img, k1, k2):
        if k1 == 0.0 and k2 == 0.0:
            return img
            
        h, w = img.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        xc, yc = w / 2.0, h / 2.0
        
        # Normalize coordinates
        x_norm = (x - xc) / xc
        y_norm = (y - yc) / yc
        r2 = x_norm**2 + y_norm**2
        
        # --- THE FIX: Viewport Scaling ---
        # Calculate the distortion at the furthest corner to find the "zoom" factor
        # needed to keep the image edges within the frame.
        r2_max = 1.0 # (at edge midpoints)
        scale_factor = 1.0 / (1.0 + k1 * r2_max + k2 * (r2_max**2))
        
        distortion = (1.0 + k1 * r2 + k2 * (r2**2)) * scale_factor
        x_dist = xc + (x - xc) * distortion
        y_dist = yc + (y - yc) * distortion
        
        return cv2.remap(img, x_dist.astype(np.float32), y_dist.astype(np.float32), 
                         cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

class GameBoyFilter(A.ImageOnlyTransform):
    """
    Simulates the original DMG-01 Game Boy display by quantizing the image into 
    4 shades of olive green.
    """
    def __init__(self, palette="green", always_apply=False, p=1.0):
        super().__init__(p=p)
        self.palette = palette
        
        # Classic DMG-01 Palette (Darkest to Lightest)
        # Note: OpenCV works in BGR by default during albumentations processing, 
        # but our dirt_pipeline expects RGB because PIL converts to RGB.
        # We define these in RGB:
        self.green_palette = np.array([
            [15, 56, 15],     # 0: Darkest Green
            [48, 98, 48],     # 1: Dark Green
            [139, 172, 15],   # 2: Light Green
            [155, 188, 15]    # 3: Lightest Green
        ], dtype=np.uint8)
        
        # Pocket / Grayscale Palette
        self.gray_palette = np.array([
            [0, 0, 0],        # 0: Black
            [85, 85, 85],     # 1: Dark Gray
            [170, 170, 170],  # 2: Light Gray
            [255, 255, 255]   # 3: White
        ], dtype=np.uint8)

    def apply(self, img, **params):
        # 1. Convert to grayscale luminance
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        else:
            gray = img.copy()
            
        # 2. Quantize into 4 bins (0, 1, 2, 3)
        bins = np.floor(gray / 64.0).astype(np.int32)
        bins = np.clip(bins, 0, 3)
        
        # 3. Map bins to the selected palette
        active_palette = self.green_palette if self.palette == "green" else self.gray_palette
        
        # Advanced NumPy indexing to map the 2D array of bin indices directly to 3D RGB colors
        quantized = active_palette[bins]
        
        return quantized

    def get_transform_init_args_names(self):
        return ("palette",)

class SmoothUpscale(A.ImageOnlyTransform):
    """
    Simulates high-quality emulator upscaling (like xBRZ/hqNx) by 
    downsampling to native resolution and upsampling via Lanczos.
    """
    def __init__(self, scale_factor=4, always_apply=False, p=1.0):
        super().__init__(p=p)
        self.scale_factor = scale_factor

    def apply(self, img, **params):
        h, w = img.shape[:2]
        # Use max(1, ...) to guarantee we never divide by zero or drop below 1 pixel
        new_w = max(1, int(w / self.scale_factor))
        new_h = max(1, int(h / self.scale_factor))
        
        # 1. Downscale to native 1x resolution (or slightly above/below)
        native = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # 2. Upscale using Lanczos4 (Smooths the edges mathematically)
        smoothed = cv2.resize(native, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return smoothed

    def get_transform_init_args_names(self):
        return ("scale_factor",)
