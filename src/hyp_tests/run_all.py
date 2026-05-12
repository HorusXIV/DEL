import os
import subprocess
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

for filename in tqdm(sorted(os.listdir(script_dir))):
    if filename.startswith("Hypothese_") and filename.endswith(".py"):
        filepath = os.path.join(script_dir, filename)
        subprocess.run(["python", filepath])
