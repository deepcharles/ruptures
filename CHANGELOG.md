# Changelog

The latest release notes are available directly in Github: [ruptures/releases](https://github.com/deepcharles/ruptures/releases).

Earlier releases are documented below ⬇️.

## [1.1.2] - 2020-12-01

### Added

- 12cbc9e feat: add piecewise linear cpd (#91)
- a12b215 test: add code coverage badge (#97)
- 2e9b17f docs: add binder for notebooks (#94)
- da7544f docs(costcosine): add entry for CostCosine in docs (#93)
- 8c9aa35 build(setup.py/cfg):  add build_ext to setup.py (#88)
- 10ef8e8 build(python39): add py39 to supported versions (#87)

### Changed

- 069bd41 fix(kernelcpd): bug fix in pelt (#95)
- b4abc34 fix: memory leak in KernelCPD (#89)

## [1.1.1] - 2020-11-26

No change to the code compared to the previous version.
The package was only partly published to Pypi because of the failure of one provider in the CI.
Since Pypi's policy prevents re-uploading twice the same version, we have to increment the version number.

## [1.1.0] - 2020-11-23

### Added

- modify publishing process to Pypi PR#83
- add cosine kernel (cost function and in KernelCPD)PR#74
- add faster kernel change point detection (`KernelCPD`, C implementation) PR#74
- add manual trigger to publish to Pypi PR#72

### Changed

## [1.0.6] - 2020-10-23
### Added

- Correct minor error in Dynp (about min_size) PR#74
- Fix legacy formatting errors PR#69
- New documentation (from Sphinx to Mkdocs) PR#64
- Separate requirements.txt and requirements-dev.txt PR#64
- A changelog file ([link](https://github.com/deepcharles/ruptures/blob/master/CHANGELOG.md))
- New Github actions for automatic generation of documentation
- Pre-commit code formatting using [black](https://github.com/psf/black)

### Changed

- Correction of display function test #64
- Add badges in the README (Github repo) PR#62: pypi version, python version, code style, contributor list
- Typo in documentation ([PR#60](https://github.com/deepcharles/ruptures/pull/60)) by @gjaeger
- Documentation theme
- Documentation site

## [1.0.5] - 2020-07-22
### Changed
- Link to documentation in PyPi description

[Unreleased]: https://github.com/deepcharles/ruptures/compare/v1.1.2...HEAD
[1.1.2]: https://github.com/deepcharles/ruptures/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/deepcharles/ruptures/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/deepcharles/ruptures/compare/v1.0.6...v1.1.0
[1.0.6]: https://github.com/deepcharles/ruptures/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/deepcharles/ruptures/compare/v1.0.4...v1.0.5
