from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
import os

class DatasetMonitor:
    def __init__(self, username: str, token: str) -> None:
        self.api = HfApi(token=token)
        self.username = username
        self.token = token

    def clean_dataset(self, data: DatasetDict):

        # -- Simple sample cleaning operations -- #
        df = data.to_pandas()
        df = df.drop_duplicates()
        df = df.dropna(how='all')
        df = df.fillna({
            'text': '', 
            'numeric_column': 0, 
        })
        # -- Include any other possible cleaning requirements as per your datasets -- #
        return Dataset.from_pandas(df) 
    
    def get_datasets_list(self):
        # -- Fetches all datasets information from the user's HuggingFace account -- #
        datasets = self.api.list_datasets(
            author=self.username,
        )
    
        for dataset in datasets:
            dataset_name = dataset.id  # dataset.id is the username/repo-name representation 
            # -- Check and skip processing dataset if already processed or not intended to be processed -- #
            if not "unprocessed" in dataset.tags or self.api.repo_exists(dataset_name + "-processed", repo_type="dataset"):
                continue
            data = load_dataset(dataset_name)
    

            # -- Each dataset has multiple splits e.g train,test -- #
            # -- We want to process them separately and maintain the data partitions -- #
            cleaned_splits = DatasetDict({
                split_name: self.clean_dataset(split_data)
                for split_name, split_data in data.items()
            })

            # -- Create a new HF repository to add the new processed dataset. -- #
            # -- Trvially, for now we are just appending -processed to symbolize it is processed repository -- #
            new_data_repo = self.api.create_repo(repo_id=dataset_name + "-processed", exist_ok=True, repo_type="dataset")
            
            # Adds the dataset to new HF dataset repository -- #
            # -- Auto-converts data from Python dictionary to Parquet format -- #
            cleaned_splits.push_to_hub(
                new_data_repo.repo_id,
                private=True,
            )

if __name__ == "__main__":
    HF_USERNAME = os.environ["HF_USERNAME"]
    HF_TOKEN = os.environ["HF_TOKEN"]
    monitor = DatasetMonitor(username=HF_USERNAME, token=HF_TOKEN)
    monitor.get_datasets_list()