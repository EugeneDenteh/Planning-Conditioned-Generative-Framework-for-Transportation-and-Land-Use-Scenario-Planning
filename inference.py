import argparse
import torch
from src.inference import semantic_maps, satellite_images
from src.records import read, seed

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--stage1", required=True)
    p.add_argument("--stage2")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    device = torch.device(a.device)
    rows = read(a.records)
    c, _ = semantic_maps(rows, a.root, a.stage1, a.output, device)
    seed(c["seed"])
    if a.stage2:
        satellite_images(rows, a.root, a.stage2, c, a.output, device)
