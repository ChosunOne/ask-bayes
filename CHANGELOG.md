# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## Releases

## 0.2.0
* Add a new wizard via `--wizard` or `-w` that guides you through updating a given hypothesis.
* Add new output formats via `--output` or `-o`.  Options are `--output json`, `--output table`, and `--output simple`.
* **BREAKING** Setting a prior for an existing hypothesis no longer requires a `--prior` or `-p` flag. E.g. `ask-bayes -n h -s -p 0.5` is now  `ask-bayes -n h -s 0.5`.

## 0.1.3
* Add additional validations to prevent P(E) from being 0.

## 0.1.2
* Add validation for inputted probabilities.
* Fix bug where you would be unable to set a non-default prior.