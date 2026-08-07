import tomllib
from pathlib import Path

from setuptools import find_packages


ROOT = Path(__file__).parents[1]


def test_setuptools_discovers_only_source_packages():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    find = config["tool"]["setuptools"]["packages"]["find"]
    assert find["namespaces"] is False
    discovered = set(
        find_packages(
            where=ROOT,
            include=find.get("include", ()),
            exclude=find.get("exclude", ()),
        )
    )
    expected = {
        ".".join(path.parent.relative_to(ROOT).parts)
        for path in (ROOT / "senpai_agent").rglob("__init__.py")
    }

    assert discovered == expected
