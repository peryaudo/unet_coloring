from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("huggan/flowers-102-categories")
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset.save_to_disk("./flowers-102-split")