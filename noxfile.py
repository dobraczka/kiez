from nox_poetry import Session, session


@session()
def tests(session: Session) -> None:
    args = session.posargs or ["--cov=kiez/", "--cov-report=xml", "tests/"]
    session.install(".[all]")
    session.install("pytest")
    session.install("pytest-cov")
    session.run("pytest", *args)


locations = ["kiez", "tests", "noxfile.py"]


@session()
def lint(session: Session) -> None:
    args = session.posargs or locations
    session.run_always("poetry", "install", external=True)
    session.run("pflake8", *args)


@session()
def type_checking(session: Session) -> None:
    args = session.posargs or locations
    session.run_always("poetry", "install", external=True)
    session.run("mypy", "--ignore-missing-imports", *args)
