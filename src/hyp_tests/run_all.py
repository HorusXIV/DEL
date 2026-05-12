import os
import runpy
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

files = [f for f in sorted(os.listdir(script_dir)) if f.startswith("Hypothese_") and f.endswith(".py")]

for filename in tqdm(files, desc="Running hypothesis scripts", unit="script"):
    filepath = os.path.join(script_dir, filename)
    try:
        tqdm.write(f"Running {filename}...")
        # execute the script in-process so GPU is shared sequentially
        runpy.run_path(filepath, run_name="__main__")
    except Exception as e:
        tqdm.write(f"Error running {filename}: {e}")
