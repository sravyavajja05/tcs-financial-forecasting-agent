from app.config import Settings


def test_mysql_url_environment_variable_configures_database(monkeypatch):
    monkeypatch.setenv(
        "MYSQL_URL",
        "mysql+pymysql://example:placeholder@localhost:3306/v2",
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.mysql_url == (
        "mysql+pymysql://example:placeholder@localhost:3306/v2"
    )
    assert settings.database_url == settings.mysql_url


def test_database_url_remains_a_compatible_environment_alias(monkeypatch):
    monkeypatch.delenv("MYSQL_URL", raising=False)
    monkeypatch.setenv(
        "DATABASE_URL",
        "sqlite:///./storage/backward-compatible.sqlite3",
    )

    settings = Settings(_env_file=None)

    assert settings.mysql_url == "sqlite:///./storage/backward-compatible.sqlite3"


def test_mysql_url_takes_precedence_over_database_url(monkeypatch):
    monkeypatch.setenv("MYSQL_URL", "sqlite:///./storage/canonical.sqlite3")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./storage/legacy.sqlite3")

    settings = Settings(_env_file=None)

    assert settings.mysql_url == "sqlite:///./storage/canonical.sqlite3"


def test_database_url_constructor_argument_still_works():
    settings = Settings(
        _env_file=None,
        database_url="sqlite:///./storage/constructor.sqlite3",
    )

    assert settings.mysql_url == "sqlite:///./storage/constructor.sqlite3"
    assert settings.database_url == settings.mysql_url
