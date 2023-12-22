from nox import session as nox_session
from nox_poetry import Session, session


@session()
def tests(session: Session) -> None:
    args = session.posargs or ["tests/"]
    session.install(".")
    session.install("pytest")
    session.install("pytest-cov")
    session.run(
        "coverage",
        "run",
        "--source=kiez",
        "--data-file=.coverage.base",
        "-m",
        "pytest",
        *args
    )


@nox_session(python="3.10", venv_backend="conda")
def test_faiss(session: Session) -> None:
    args = session.posargs or ["tests/"]
    session.conda_install(
        "-c", "pytorch", "faiss-cpu=1.7.4", "mkl=2021", "blas=1.0=mkl"
    )
    session.install(".")
    session.install("autofaiss")
    session.install("pytest")
    session.install("pytest-cov")
    session.run(
        "coverage",
        "run",
        "--source=kiez",
        "--data-file=.coverage.faiss",
        "-m",
        "pytest",
        *args
    )


@session(python="3.10")
def test_ngt(session: Session) -> None:
    args = session.posargs or ["tests/"]
    session.install(".[ngt]")
    session.install("pytest")
    session.install("pytest-cov")
    session.run(
        "coverage",
        "run",
        "--source=kiez",
        "--data-file=.coverage.ngt",
        "-m",
        "pytest",
        *args
    )


@session(python="3.10")
def test_nmslib(session: Session) -> None:
    args = session.posargs or ["tests/"]
    session.install(".[nmslib]")
    session.install("pytest")
    session.install("pytest-cov")
    session.run(
        "coverage",
        "run",
        "--source=kiez",
        "--data-file=.coverage.nmslib",
        "-m",
        "pytest",
        *args
    )


@session(python="3.10")
def test_annoy(session: Session) -> None:
    args = session.posargs or ["tests/"]
    session.install(".[annoy]")
    session.install("pytest")
    session.install("pytest-cov")
    session.run(
        "coverage",
        "run",
        "--source=kiez",
        "--data-file=.coverage.annoy",
        "-m",
        "pytest",
        *args
    )


locations = ["kiez", "tests", "noxfile.py"]


@session()
def lint(session: Session) -> None:
    args = session.posargs or locations
    session.install("black", "isort")
    session.run("black", *args)
    session.run("isort", *args)


@session()
def style_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install(
        "pyproject-flake8",
        "flake8-eradicate",
        "flake8-isort",
        "flake8-debugger",
        "flake8-comprehensions",
        "flake8-print",
        "flake8-black",
        "flake8-bugbear",
        "pydocstyle",
    )
    session.run("pflake8", *args)


@session()
def pyroma(session: Session) -> None:
    session.install("pyroma")
    session.run("pyroma", "--min", "10", ".")


@session()
def type_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install("mypy")
    session.run("mypy", "--ignore-missing-imports", *args)


@session()
def build_docs(session: Session) -> None:
    session.install(".")
    session.install("sphinx")
    session.install("insegel")
    session.cd("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
