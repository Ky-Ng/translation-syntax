"""Shared HuggingFace Hub helpers.

Per CLAUDE.md: datasets are public and belong to the per-experiment collection
`translation-syntax-datasets-<EXPERIMENT_NUMBER>`.
"""

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
COLLECTION_TITLE = f"translation-syntax-datasets-{EXPERIMENT_DIR.name}"


def push_dataset(repo_id, files, card_text):
    """Upload `files` ({path_in_repo: local_path}) plus a README card to a
    public dataset repo, and ensure it is in the experiment's collection."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_settings(repo_id, repo_type="dataset", private=False)
    for path_in_repo, local_path in files.items():
        api.upload_file(
            path_or_fileobj=str(local_path), path_in_repo=path_in_repo, repo_id=repo_id, repo_type="dataset"
        )
    api.upload_file(path_or_fileobj=card_text.encode(), path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")

    owner = repo_id.split("/")[0]
    collection = next((c for c in api.list_collections(owner=[owner]) if c.title == COLLECTION_TITLE), None)
    if collection is None:
        collection = api.create_collection(COLLECTION_TITLE)
    api.add_collection_item(collection.slug, item_id=repo_id, item_type="dataset", exists_ok=True)
    return f"https://huggingface.co/datasets/{repo_id}"
