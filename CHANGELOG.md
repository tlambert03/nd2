# Changelog

## [0.4.6](https://github.com/tlambert03/nd2/tree/0.4.6) (2022-10-08)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.4.5...0.4.6)

**Implemented enhancements:**

- feat: add roi parsing [\#102](https://github.com/tlambert03/nd2/pull/102) ([tlambert03](https://github.com/tlambert03))

## [v0.4.5](https://github.com/tlambert03/nd2/tree/v0.4.5) (2022-10-06)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.4.4...v0.4.5)

**Implemented enhancements:**

- feat: offer full unstructured data [\#101](https://github.com/tlambert03/nd2/pull/101) ([tlambert03](https://github.com/tlambert03))

**Fixed bugs:**

- fix: fix rounding error in numpy pre-release [\#96](https://github.com/tlambert03/nd2/pull/96) ([tlambert03](https://github.com/tlambert03))

**Merged pull requests:**

- ci\(dependabot\): bump pypa/cibuildwheel from 2.10.1 to 2.10.2 [\#98](https://github.com/tlambert03/nd2/pull/98) ([dependabot[bot]](https://github.com/apps/dependabot))
- ci\(dependabot\): bump pypa/cibuildwheel from 2.9.0 to 2.10.1 [\#97](https://github.com/tlambert03/nd2/pull/97) ([dependabot[bot]](https://github.com/apps/dependabot))

## [v0.4.4](https://github.com/tlambert03/nd2/tree/v0.4.4) (2022-09-12)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.4.3...v0.4.4)

**Fixed bugs:**

- fix: fix image width with sdk reader [\#94](https://github.com/tlambert03/nd2/pull/94) ([tlambert03](https://github.com/tlambert03))

## [v0.4.3](https://github.com/tlambert03/nd2/tree/v0.4.3) (2022-09-11)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.4.2...v0.4.3)

**Implemented enhancements:**

- Add `unstructured_metadata` method, fix missing position names in experiment. [\#93](https://github.com/tlambert03/nd2/pull/93) ([tlambert03](https://github.com/tlambert03))

## [v0.4.2](https://github.com/tlambert03/nd2/tree/v0.4.2) (2022-09-10)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.4.1...v0.4.2)

**Fixed bugs:**

- fix: better fix for images with non-normal strides [\#92](https://github.com/tlambert03/nd2/pull/92) ([tlambert03](https://github.com/tlambert03))

## [v0.4.1](https://github.com/tlambert03/nd2/tree/v0.4.1) (2022-09-09)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.4.0...v0.4.1)

**Fixed bugs:**

- fix: Fix images where widthPx x Bytes is not the same as expected width Bytes [\#90](https://github.com/tlambert03/nd2/pull/90) ([tlambert03](https://github.com/tlambert03))

## [v0.4.0](https://github.com/tlambert03/nd2/tree/v0.4.0) (2022-08-19)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.3.0...v0.4.0)

**Implemented enhancements:**

- feat: Add native macosx-arm64 \(M1\) support [\#87](https://github.com/tlambert03/nd2/pull/87) ([tlambert03](https://github.com/tlambert03))

**Fixed bugs:**

- fix: Typo in legacy requirements [\#79](https://github.com/tlambert03/nd2/pull/79) ([c0deizbae](https://github.com/c0deizbae))

**Merged pull requests:**

- ci\(dependabot\): bump pypa/cibuildwheel from 2.8.1 to 2.9.0 [\#84](https://github.com/tlambert03/nd2/pull/84) ([dependabot[bot]](https://github.com/apps/dependabot))
- ci\(dependabot\): bump pypa/cibuildwheel from 2.8.0 to 2.8.1 [\#81](https://github.com/tlambert03/nd2/pull/81) ([dependabot[bot]](https://github.com/apps/dependabot))
- ci\(dependabot\): bump pypa/cibuildwheel from 2.7.0 to 2.8.0 [\#80](https://github.com/tlambert03/nd2/pull/80) ([dependabot[bot]](https://github.com/apps/dependabot))

## [v0.3.0](https://github.com/tlambert03/nd2/tree/v0.3.0) (2022-06-25)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.2.5...v0.3.0)

**Implemented enhancements:**

- feat: add `read_using_sdk` parameter, default to True for compressed files [\#74](https://github.com/tlambert03/nd2/pull/74) ([tlambert03](https://github.com/tlambert03))

**Documentation:**

- docs: update readme [\#66](https://github.com/tlambert03/nd2/pull/66) ([tlambert03](https://github.com/tlambert03))

**Merged pull requests:**

- ci\(dependabot\): bump pypa/cibuildwheel from 2.6.1 to 2.7.0 [\#69](https://github.com/tlambert03/nd2/pull/69) ([dependabot[bot]](https://github.com/apps/dependabot))
- ci\(dependabot\): bump pypa/cibuildwheel from 2.6.0 to 2.6.1 [\#68](https://github.com/tlambert03/nd2/pull/68) ([dependabot[bot]](https://github.com/apps/dependabot))
- ci\(dependabot\): bump actions/setup-python from 3 to 4 [\#67](https://github.com/tlambert03/nd2/pull/67) ([dependabot[bot]](https://github.com/apps/dependabot))

## [v0.2.5](https://github.com/tlambert03/nd2/tree/v0.2.5) (2022-06-06)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.2.4...v0.2.5)

**Fixed bugs:**

- fix: remove print statement [\#65](https://github.com/tlambert03/nd2/pull/65) ([tlambert03](https://github.com/tlambert03))
- fix: fix pre-test by using PyArray\_SetBaseObject directly [\#64](https://github.com/tlambert03/nd2/pull/64) ([tlambert03](https://github.com/tlambert03))

**Tests & CI:**

- ci: use flake8-pyprojecttoml [\#63](https://github.com/tlambert03/nd2/pull/63) ([tlambert03](https://github.com/tlambert03))
- ci: create dependabot.yml [\#59](https://github.com/tlambert03/nd2/pull/59) ([tlambert03](https://github.com/tlambert03))

**Merged pull requests:**

- ci: update prefix on dependabot commit message [\#61](https://github.com/tlambert03/nd2/pull/61) ([tlambert03](https://github.com/tlambert03))
- ci\(dependabot\): Bump codecov/codecov-action from 2 to 3 [\#60](https://github.com/tlambert03/nd2/pull/60) ([dependabot[bot]](https://github.com/apps/dependabot))
- ci: General repo update, move to pyproject.toml [\#58](https://github.com/tlambert03/nd2/pull/58) ([tlambert03](https://github.com/tlambert03))
- build: remove extra numpy from install\_requires [\#57](https://github.com/tlambert03/nd2/pull/57) ([tlambert03](https://github.com/tlambert03))

## [v0.2.4](https://github.com/tlambert03/nd2/tree/v0.2.4) (2022-05-26)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.2.3...v0.2.4)

**Fixed bugs:**

- fix numpy 1.22.4 bug [\#55](https://github.com/tlambert03/nd2/pull/55) ([tlambert03](https://github.com/tlambert03))
- Fix frame offsets when validate\_frames != True, add `validate_frames` param to `imread` [\#54](https://github.com/tlambert03/nd2/pull/54) ([tlambert03](https://github.com/tlambert03))

## [v0.2.3](https://github.com/tlambert03/nd2/tree/v0.2.3) (2022-05-18)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.2.2...v0.2.3)

**Fixed bugs:**

- Refactor fixup [\#51](https://github.com/tlambert03/nd2/pull/51) ([shenker](https://github.com/shenker))

## [v0.2.2](https://github.com/tlambert03/nd2/tree/v0.2.2) (2022-03-13)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.2.1...v0.2.2)

**Implemented enhancements:**

- Add utility to try to rescue frames from a corrupted nd2 file [\#44](https://github.com/tlambert03/nd2/pull/44) ([tlambert03](https://github.com/tlambert03))

**Fixed bugs:**

- Fix numpy pinning in wheels [\#46](https://github.com/tlambert03/nd2/pull/46) ([tlambert03](https://github.com/tlambert03))

## [v0.2.1](https://github.com/tlambert03/nd2/tree/v0.2.1) (2022-03-02)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.2.0...v0.2.1)

**Merged pull requests:**

- update SDK license [\#43](https://github.com/tlambert03/nd2/pull/43) ([tlambert03](https://github.com/tlambert03))

## [v0.2.0](https://github.com/tlambert03/nd2/tree/v0.2.0) (2022-02-20)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.1.5-1...v0.2.0)

**Merged pull requests:**

- add gh release action [\#42](https://github.com/tlambert03/nd2/pull/42) ([tlambert03](https://github.com/tlambert03))
- make 2 utils public [\#41](https://github.com/tlambert03/nd2/pull/41) ([tlambert03](https://github.com/tlambert03))
- Use resource-backed-dask-array, copy dask chunks by default [\#40](https://github.com/tlambert03/nd2/pull/40) ([tlambert03](https://github.com/tlambert03))
- \[pre-commit.ci\] pre-commit autoupdate [\#37](https://github.com/tlambert03/nd2/pull/37) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- \[pre-commit.ci\] pre-commit autoupdate [\#36](https://github.com/tlambert03/nd2/pull/36) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- \[pre-commit.ci\] pre-commit autoupdate [\#35](https://github.com/tlambert03/nd2/pull/35) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- \[pre-commit.ci\] pre-commit autoupdate [\#34](https://github.com/tlambert03/nd2/pull/34) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- \[pre-commit.ci\] pre-commit autoupdate [\#33](https://github.com/tlambert03/nd2/pull/33) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))

## [v0.1.5-1](https://github.com/tlambert03/nd2/tree/v0.1.5-1) (2021-11-12)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.1.6...v0.1.5-1)

## [v0.1.6](https://github.com/tlambert03/nd2/tree/v0.1.6) (2021-11-12)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.1.5...v0.1.6)

## [v0.1.5](https://github.com/tlambert03/nd2/tree/v0.1.5) (2021-11-12)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.1.4...v0.1.5)

**Merged pull requests:**

- Small update to ND2File.sizes, expose `frame_metadata` method [\#32](https://github.com/tlambert03/nd2/pull/32) ([tlambert03](https://github.com/tlambert03))
- Add check-manifest during deploy [\#31](https://github.com/tlambert03/nd2/pull/31) ([tlambert03](https://github.com/tlambert03))
- move sdk files into arch subfolders [\#29](https://github.com/tlambert03/nd2/pull/29) ([tlambert03](https://github.com/tlambert03))
- make test dataset easier to get [\#28](https://github.com/tlambert03/nd2/pull/28) ([tlambert03](https://github.com/tlambert03))
- Add pickle test [\#27](https://github.com/tlambert03/nd2/pull/27) ([tlambert03](https://github.com/tlambert03))
- Better dask wrapper that obeys NEP18 [\#26](https://github.com/tlambert03/nd2/pull/26) ([tlambert03](https://github.com/tlambert03))
- Add support for python 3.10 [\#23](https://github.com/tlambert03/nd2/pull/23) ([tlambert03](https://github.com/tlambert03))
- add Dask-proxy [\#22](https://github.com/tlambert03/nd2/pull/22) ([tlambert03](https://github.com/tlambert03))
- Include SDK, move to src layout [\#21](https://github.com/tlambert03/nd2/pull/21) ([tlambert03](https://github.com/tlambert03))
- Add \_\_getstate\_\_ and \_\_setstate\_\_ methods to allow pickling. Addresses \#19 [\#20](https://github.com/tlambert03/nd2/pull/20) ([VolkerH](https://github.com/VolkerH))
- \[pre-commit.ci\] pre-commit autoupdate [\#13](https://github.com/tlambert03/nd2/pull/13) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- fix coverage [\#12](https://github.com/tlambert03/nd2/pull/12) ([tlambert03](https://github.com/tlambert03))

## [v0.1.4](https://github.com/tlambert03/nd2/tree/v0.1.4) (2021-10-10)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.1.3...v0.1.4)

**Merged pull requests:**

- allow position arg to asarray [\#11](https://github.com/tlambert03/nd2/pull/11) ([tlambert03](https://github.com/tlambert03))

## [v0.1.3](https://github.com/tlambert03/nd2/tree/v0.1.3) (2021-10-10)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.1.2...v0.1.3)

**Merged pull requests:**

- Fix-legacy, put more in base readers [\#10](https://github.com/tlambert03/nd2/pull/10) ([tlambert03](https://github.com/tlambert03))

## [v0.1.2](https://github.com/tlambert03/nd2/tree/v0.1.2) (2021-10-09)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.1.1...v0.1.2)

**Merged pull requests:**

- add copy param to to\_dask to avoid segfaults from closed file [\#9](https://github.com/tlambert03/nd2/pull/9) ([tlambert03](https://github.com/tlambert03))

## [v0.1.1](https://github.com/tlambert03/nd2/tree/v0.1.1) (2021-10-07)

[Full Changelog](https://github.com/tlambert03/nd2/compare/v0.1.0...v0.1.1)

**Merged pull requests:**

- optional squeeze param for xarray [\#8](https://github.com/tlambert03/nd2/pull/8) ([tlambert03](https://github.com/tlambert03))
- add test that readme executes [\#7](https://github.com/tlambert03/nd2/pull/7) ([tlambert03](https://github.com/tlambert03))

## [v0.1.0](https://github.com/tlambert03/nd2/tree/v0.1.0) (2021-10-07)

[Full Changelog](https://github.com/tlambert03/nd2/compare/2c209d5857873d44345a9143a8ec8c3c3b3a0c76...v0.1.0)

**Merged pull requests:**

- readme updates [\#6](https://github.com/tlambert03/nd2/pull/6) ([tlambert03](https://github.com/tlambert03))
- general cleanup, full legacy support [\#5](https://github.com/tlambert03/nd2/pull/5) ([tlambert03](https://github.com/tlambert03))
- add support for legacy SDK [\#2](https://github.com/tlambert03/nd2/pull/2) ([tlambert03](https://github.com/tlambert03))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
