import traceback
import time
from pathlib import Path
import cv2
import fire
import pandas as pd
# THE FIX: Swap thread_map for process_map to bypass GIL
from tqdm.contrib.concurrent import process_map 

from manga_ocr_dev.env import FONTS_ROOT, DATA_SYNTHETIC_ROOT
from manga_ocr_dev.synthetic_data_generator.retro_generator import RetroGenerator

def f(args):
    try:
        # We ignore passed 'line' from CSV and generate our own via RetroGenerator
        i, source, id_, _, out_dir = args 
        
        # Instantiate locally inside the process to avoid memory sharing issues
        gen = RetroGenerator() 
        
        filename = f"{id_}.jpg"
        
        # Generate image, text and font path
        img, text_gt, font_path = gen.generate_one()
        
        # Save
        cv2.imwrite(str(out_dir / filename), img)
        
        # Extract relative font filename for metadata
        font_filename = Path(font_path).name
        
        return source, id_, text_gt, False, font_filename

    except Exception:
        print(traceback.format_exc())
        return None

def run(package=0, n_random=1000, n_limit=None, max_workers=12):
    """
    :param package: number of data package to generate
    :param n_random: how many samples with random text to generate
    :param n_limit: limit number of generated samples (for debugging)
    :param max_workers: max number of workers
    """
    package_str = f"{package:04d}"
    out_dir = DATA_SYNTHETIC_ROOT / "img" / package_str
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup dummy args for the random generation loop
    # We use a dataframe as the backbone to maintain compatibility with MangaOCR meta structure
    data_args = []
    for i in range(n_random):
        data_args.append((i, "retro_synthetic", f"retro_{package_str}_{i}", None, out_dir))
    
    if n_limit:
        data_args = data_args[:n_limit]

    print(f"Starting multiprocessing generation for package {package_str}...")
    
    # THE FIX: True Multiprocessing with chunksize optimization
    data = process_map(f, data_args, max_workers=max_workers, chunksize=10, desc=f"Generating {package_str}")

    # Filter out any None results from exceptions
    data = [d for d in data if d is not None]

    data_df = pd.DataFrame(data, columns=["source", "id", "text", "vertical", "font_path"])
    meta_path = DATA_SYNTHETIC_ROOT / f"meta/{package_str}.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(meta_path, index=False)
    print(f"Package {package_str} complete. Metadata saved to {meta_path}")

if __name__ == "__main__":
    fire.Fire(run)
