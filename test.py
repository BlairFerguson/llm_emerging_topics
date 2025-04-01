from datasets import load_dataset

dataset = load_dataset(
    "PlanTL-GOB-ES/SQAC",
    data_files={"train": "train.json", "test": "test.json", "dev": "dev.json"},
    split="train",
    trust_remote_code=True  # Allows execution of the dataset's custom loading script
)

print(dataset)  # Print dataset details to confirm it loads
