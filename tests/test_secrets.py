from senpai_agent.secrets import scrub_github_credentials


def test_scrub_github_credentials_removes_every_handoff():
    environment = {
        "GITHUB_TOKEN": "token",
        "GH_TOKEN": "token",
        "SENPAI_GITHUB_TOKEN_FILE": "/secret",
        "SENPAI_GITHUB_TOKEN_FD": "47",
        "WANDB_API_KEY": "keep",
    }

    scrub_github_credentials(environment)

    assert environment == {"WANDB_API_KEY": "keep"}
