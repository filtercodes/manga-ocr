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

# Global cache for the generator to avoid re-initialization in worker processes
_worker_gen = None

def f(args):
    global _worker_gen
    try:
        # We ignore passed 'line' from CSV and generate our own via RetroGenerator
        i, source, id_, _, out_dir = args 
        
        # Instantiate locally inside the process once to avoid overhead
        if _worker_gen is None:
            _worker_gen = RetroGenerator() 
        
        filename = f"{id_}.jpg"
        
        # Catch the 4th return value (debug_dict) for observability
        img, text_gt, font_path, debug_dict = _worker_gen.generate_one()
        
        # Save
        cv2.imwrite(str(out_dir / filename), img)
        
        # Extract relative font filename for metadata
        font_filename = Path(font_path).name
        
        # Return a flat tuple containing standard data PLUS the debug fields
        return (
            source, id_, text_gt, False, font_filename,
            debug_dict["category"],
            debug_dict["archetype"],
            debug_dict["bg_luminance"],
            debug_dict["bg_noise"],
            debug_dict["fg_color"],
            debug_dict["outline_color"],
            debug_dict["outline_iters"],
            debug_dict["outline_shape"],
            debug_dict["has_shadow"],
            debug_dict["pipeline_used"]
        )

    except Exception:
        print(traceback.format_exc())
        return None

def run(package=None, n_random=1000, n_limit=None, max_workers=12):
    """
    :param package: specific package number (auto-increments if None)
    :param n_random: how many samples with random text to generate
    :param n_limit: limit number of generated samples (for debugging)
    :param max_workers: max number of workers
    """
    # Auto-increment logic
    if package is None:
        package = 0
        while (DATA_SYNTHETIC_ROOT / "img" / f"{package:04d}").exists():
            package += 1
            
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
    start_time = time.time()
    
    # THE FIX: True Multiprocessing with chunksize 1 for smooth UI and zippy performance
    data = process_map(f, data_args, max_workers=max_workers, chunksize=1, desc=f"Generating {package_str}")

    # Filter out any None results from exceptions
    data = [d for d in data if d is not None]

    # Expand columns to include telemetry metadata
    columns = [
        "source", "id", "text", "vertical", "font_path", 
        "debug_category", "debug_archetype", "debug_luminance", "debug_noise",
        "debug_fg_color", "debug_outline_color", "debug_outline_iters", 
        "debug_outline_shape", "debug_shadow", "debug_pipeline"
    ]
    
    data_df = pd.DataFrame(data, columns=columns)
    meta_path = DATA_SYNTHETIC_ROOT / f"meta/{package_str}.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(meta_path, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Package {package_str} complete.")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average speed: {len(data)/elapsed_time:.2f} images/sec")
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    fire.Fire(run)
