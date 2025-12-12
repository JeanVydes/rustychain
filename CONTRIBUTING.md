# Contributing to RustyChain

Thank you for your interest in contributing!

## How to Contribute

1. **Fork the repository** and create your branch from `main`.
2. **Write clear, well-documented code** and include tests when possible.
3. **Run all tests** before submitting a pull request.
4. **Open a pull request** with a clear description of your changes.
5. Be respectful and constructive in code reviews and discussions.

## Code Style
- Follow Rust best practices and formatting (`cargo fmt`).
- Document public APIs with doc comments.

## Local checks (pre-commit / pre-push)

Before opening a PR, run the following checks locally to match CI expectations:

- Run Clippy with automatic fixes (may edit files):

```bash
# If your workspace is a git repo use:
cargo clippy --fix --lib -p rustychain --all-features --allow-dirty

# If you're running in an environment without a VCS (or CI container), add:
cargo clippy --fix --lib -p rustychain --all-features --allow-dirty --allow-no-vcs
```

- Generate docs (CI runs documentation checks as well):

```bash
cargo doc --all-features --no-deps --document-private-items
```

- Check formatting (CI runs `cargo fmt --all -- --check`):

```bash
cargo fmt --check --all
```

These commands mirror the project's GitHub Actions (`.github/workflows/rust.yml`) so running them locally helps catch issues before creating a PR.

## Running tests

Run the unit tests and integration tests locally. Some integration tests require a Postgres instance with `pgvector` installed — the CI provides a service for that. Locally you can either run Postgres (with `pgvector`) or rely on `testcontainers` in tests/benches that spawn ephemeral containers.

- Run library unit tests (all features):

```bash
cargo test --lib --all-features
```

- Run doctests:

```bash
cargo test --doc --all-features
```

- Run integration tests (if you have a local Postgres with `pgvector`):

```bash
export DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/test_db
cargo test --test '*' --all-features
```

- If you don't have a local `pgvector` Postgres, the crate's benches/tests may use `testcontainers` to start a container. Ensure Docker is running for those tests to succeed.

## Reporting Issues
- Use the issue tracker for bugs, feature requests, or questions.
- Provide as much detail as possible (logs, OS, steps to reproduce).

## Code of Conduct
- Be kind, inclusive, and respectful to others.

---

Happy coding!
