from dataset import load_split

if __name__ == "__main__":
    # Pinned dataset revision and fixed seed (see dataset.load_split), so the split is reproducible.
    dataset = load_split()
    dataset.save_to_disk("./flowers-102-split")
