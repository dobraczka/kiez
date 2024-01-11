# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.5.0] - 2024-01-11

### Added

- Support for torch, when using Faiss
- More metrics for Faiss

### Changed

- Simplified kneighbors API, i.e. no queries can be supplied anymore, since they need to come from the source anyway

### Removed

- Autofaiss support was removed

### Fixed

- Several efficiency problems when not using Hubness Reduction were addressed

## [0.4.4] - 2023-10-18

### Changed

- Loosen class-resolver dependency
- Simplify dependencies for extras

### Fixed

- Avoid computing initial kcandidates when not using HubnessReduction

## [0.4.3] - 2023-03-27

### Fixed

- Loosen python dependency restriction


## [0.4.2] - 2022-11-09

### Fixed

- Upgrade class-resolver dependency

## [0.4.1] - 2022-10-12

### Fixed

- Fix joblib vulnerability by raising minimum required version

## [0.4.0] - 2022-06-01

### Added

- Added Faiss to enable fast hubness-reduced nearest neighbor search on the gpu

## [0.3.3] - 2022-02-08

### Fixed

- Patch IPython vulnerability by setting it to >=7.31.1

## [0.3.2] - 2022-01-06

### Fixed

- Fixed some version constraints of dependencies

## [0.3.1] - 2021-11-23

### Fixed

- Relaxed some version constraints of dependencies

### Added

- Enhanced documentation and some type hint problems


## [0.3.0] - 2021-08-09

### Added

- More possibilities to instantiate Kiez (thanks to @cthoyt )
- Single-source use simplified
- Enhanced documentation

## [0.2.2] - 2021-07-22

### Fixed

- Fix minor error in README and licensing for PyPI

## [0.2.1] - 2021-07-22

### Fixed

- Fix problems with readthedocs

[0.4.4]: https://github.com/dobraczka/kiez/releases/tag/0.4.4
[0.4.3]: https://github.com/dobraczka/kiez/releases/tag/0.4.3
[0.4.2]: https://github.com/dobraczka/kiez/releases/tag/0.4.2
[0.4.1]: https://github.com/dobraczka/kiez/releases/tag/0.4.1
[0.4.0]: https://github.com/dobraczka/kiez/releases/tag/0.4.0
[0.3.3]: https://github.com/dobraczka/kiez/releases/tag/0.3.3
[0.3.2]: https://github.com/dobraczka/kiez/releases/tag/0.3.2
[0.3.1]: https://github.com/dobraczka/kiez/releases/tag/0.3.1
[0.3.0]: https://github.com/dobraczka/kiez/releases/tag/0.3.0
[0.2.2]: https://github.com/dobraczka/kiez/releases/tag/0.2.2
[0.2.1]: https://github.com/dobraczka/kiez/releases/tag/0.2.1
