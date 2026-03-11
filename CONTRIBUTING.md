# Contributing to .aFix

Thank you for your interest in contributing to the .aFix Protocol!

## Ways to Contribute

- **Specification feedback** — Open an issue to propose changes to `SPEC.md`.
- **Encoder/Decoder** — See `src/` for the Rust and C++ implementations.
- **Web Component** — See `tools/web/` for the `afix-view` component.
- **Documentation** — See `docs/` for guides and architecture deep-dives.

## Development Process

1. Fork the repository and create a feature branch.
2. Follow the coding conventions outlined in `docs/` for the relevant component.
3. Ensure all tests pass (`cargo test` for Rust, `npm test` for JS).
4. Open a pull request with a clear description of your change.

## Specification Changes

Changes to the binary specification (breaking `MAJOR` bumps) require:

- An RFC (Request for Comments) issue opened at least 30 days before implementation.
- Sign-off from two core maintainers.
- A corresponding update to `SPEC.md` and `spec/` directory.

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct.

## Licence

By contributing, you agree that your contributions will be licenced under the [MIT Licence](./LICENSE).
