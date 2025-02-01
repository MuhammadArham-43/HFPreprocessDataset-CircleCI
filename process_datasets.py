from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
import os

HF_USERNAME = os.environ["HF_USERNAME"]
HF_TOKEN = os.environ["HF_TOKEN"]

class DatasetMonitor:
    def __init__(self, username: str, token: str) -> None:
        self.api = HfApi(token=token)
        self.username = username
        self.token = token

    def clean_dataset(self, data: DatasetDict):
        df = data.to_pandas()
        df = df.drop_duplicates()
        df = df.dropna(how='all')
        df = df.fillna({
            'text': '', 
            'numeric_column': 0, 
        })
        # -- Other possible cleaning requirements as per your datasets -- #
        return Dataset.from_pandas(df) 
    
    def get_datasets_list(self):
        datasets = self.api.list_datasets(
            author=self.username,
        )
    
        for dataset in datasets:
            dataset_name = dataset.id
            if not "unprocessed" in dataset.tags or self.api.repo_exists(dataset_name + "-processed", repo_type="dataset"):
                continue
    
            data = load_dataset(dataset_name)
            cleaned_splits = DatasetDict({
                split_name: self.clean_dataset(split_data)
                for split_name, split_data in data.items()
            })
    
            new_data_repo = self.api.create_repo(repo_id=dataset_name + "-processed", exist_ok=True, repo_type="dataset")
            # Auto-converts the data to Parquet format
            cleaned_splits.push_to_hub(
                new_data_repo.repo_id,
                private=True,
            )

if __name__ == "__main__":
    monitor = DatasetMonitor(username=HF_USERNAME, token=HF_TOKEN)
    monitor.get_datasets_list()