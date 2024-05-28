import argparse
import os
import shutil
from huggingface_hub import HfApi, Repository

hf_username = 'haining'

def upload_model_to_hf(hf_username, hf_repo_name, model_checkpoint_dir):
    # create a new repository on Hugging Face Hub
    api = HfApi()
    api.create_repo(repo_id=f"{hf_username}/{hf_repo_name}", private=False)

    # clone the repository
    repo = Repository(local_dir=f"./{hf_repo_name}",
                      clone_from=f"{hf_username}/{hf_repo_name}")

    # copy your model files to the repository directory
    for file_name in os.listdir(model_checkpoint_dir):
        full_file_name = os.path.join(model_checkpoint_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, repo.local_dir)

    # push the files to the repository
    repo.push_to_hub()

    print(f"Model {hf_repo_name} has been successfully uploaded to Hugging Face Hub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload model checkpoint to Hugging Face Hub.")
    parser.add_argument('--repo_name', type=str, required=True,
                        help="The name of the Hugging Face repository.")
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help="The directory of the model checkpoint.")

    args = parser.parse_args()

    upload_model_to_hf(args.hf_username, args.hf_repo_name, args.model_checkpoint_dir)
