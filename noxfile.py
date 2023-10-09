from nox_poetry import Session, session


@session(tags=["test"])
def tests(session: Session) -> None:
    args = session.posargs or ["--cov=kiez/", "--cov-report=xml", "tests/"]
    session.install(".[all]")
    session.install("pytest")
    session.install("pytest-cov")
    session.run("pytest", *args)


locations = ["kiez", "tests", "noxfile.py"]


@session(tags=["fix"])
def lint(session: Session) -> None:
    args = session.posargs or locations
    session.install("black", "isort")
    session.run("black", *args)
    session.run("isort", *args)


@session(tags=["style"])
def style_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install(
        "pyproject-flake8",
        "flake8-eradicate",
        "flake8-isort",
        "flake8-debugger",
        "flake8-comprehensions",
        "flake8-print",
    )
    session.run("pflake8", *args)


@session(tags=["style"])
def pyroma(session: Session) -> None:
    session.install("pyroma")
    session.run("pyroma", "--min", "10", ".")


@session(tags=["style", "docs"])
def doctests(session: Session) -> None:
    session.install(".[all]")
    session.install("xdoctest")
    session.install("pygments")
    session.run("xdoctest", "-m", "kiez")


@session(tags=["style"])
def type_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install("mypy")
    session.run("mypy", "--ignore-missing-imports", *args)


@session(tags=["docs"])
def build_docs(session: Session) -> None:
    session.install(".")
    session.install("sphinx")
    session.install("insegel")
    session.cd("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
