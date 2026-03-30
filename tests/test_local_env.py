"""local_env: .env 読み込み（git 管理外ファイル）。"""

import os

from experiments.local_env import load_repo_dotenv


def test_load_repo_dotenv_sets_missing_only(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        'OPENAI_API_KEY=from_file\n'
        "# comment\n"
        'OTHER=2\n',
        encoding="utf-8",
    )
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        os.environ["OPENAI_API_KEY"] = "already_set"
        load_repo_dotenv(repo_root=tmp_path)
        assert os.environ["OPENAI_API_KEY"] == "already_set"
        assert os.environ.get("OTHER") == "2"
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old
        else:
            os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OTHER", None)


def test_load_repo_dotenv_fills_when_unset(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=secret_from_env_file\n",
        encoding="utf-8",
    )
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        load_repo_dotenv(repo_root=tmp_path)
        assert os.environ["OPENAI_API_KEY"] == "secret_from_env_file"
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old
        else:
            os.environ.pop("OPENAI_API_KEY", None)
