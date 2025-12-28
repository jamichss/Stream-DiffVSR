from huggingface_hub import create_repo, delete_repo, HfApi

create_repo("Jamichsu/Stream-DiffVSR", private = False)

api = HfApi()

api.upload_folder(
    folder_path="pretrained_model",
    # path_in_repo="reds_ckpt",
    repo_id="Jamichsu/Stream-DiffVSR",
    repo_type="model",
    # ignore_patterns="**/logs/*.txt",
)