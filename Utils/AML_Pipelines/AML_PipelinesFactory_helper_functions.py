import git
import os


def get_git_properties() -> str:
    """Returns the git properties in a string"""
    repo = git.Repo(os.getcwd(), search_parent_directories=True)
    branch_name = repo.active_branch.name
    commit_hash = repo.head.object.hexsha
    return f"Git branch: {branch_name}\nGit commit: {commit_hash}"
