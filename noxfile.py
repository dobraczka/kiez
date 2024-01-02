from nox import session as nox_session
from nox_poetry import Session, session


@session(tags=["tests"])
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
        *args,
    )


@nox_session(python="3.10", venv_backend="conda", tags=["tests"])
def test_faiss(session: Session) -> None:
    args = session.posargs or ["tests/"]
    session.conda_install(
        "-c", "pytorch", "faiss-cpu=1.7.4", "mkl=2021", "blas=1.0=mkl"
    )
    session.conda_install("-c", "pytorch", "pytorch=2.1.2", "cpuonly")
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
        *args,
    )


@session(python="3.10", tags=["tests"])
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
        *args,
    )


@session(python="3.10", tags=["tests"])
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
        *args,
    )


@session(python="3.10", tags=["tests"])
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
        *args,
    )


@session(python="3.10", tags=["tests"])
def coverage(session: Session) -> None:
    session.install("pytest")
    session.install("pytest-cov")
    session.run("coverage", "combine")
    session.run("coverage", "html")


locations = ["kiez", "tests", "noxfile.py"]


@session()
def lint(session: Session) -> None:
    session.install("pre-commit")
    session.run(
        "pre-commit",
        "run",
        "--all-files",
        "--hook-stage=manual",
        *session.posargs,
    )


@session()
def style_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install("ruff")
    session.run("ruff", "check", *args)


@session()
def pedantic_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install("ruff")
    session.run("ruff", "check", '--extend-select="ARG,TID,PLR0913,PLR0912"', *args)


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
