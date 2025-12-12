# Benchmarks — rustychain

This document explains how to run the benchmarks included in the `rustychain` crate, with practical notes and troubleshooting tips.

## Prerequisites

- Docker running locally (required by testcontainers).
- Rust toolchain (stable) and Cargo.
- `cargo` dev-dependencies are declared in the workspace (`testcontainers`, `criterion`, etc.).
- If you prefer to use an existing Postgres instead of Docker, that Postgres must have the `pgvector` extension installed.

## Benchmarks in this crate

- `vector_store` — benchmark for the `VectorStore` implementation (adds documents and runs similarity searches). This bench uses `testcontainers` to spawn an ephemeral PostgreSQL container with `pgvector` available.

The `vector_store` bench is defined in `benches/vector_store.rs`.

## Recommended command (single bench)

Run the `vector_store` bench (this example enables features required for Postgres support and optional providers used by the crate):

```bash
cargo bench -p rustychain --features "postgres providers" --bench vector_store -- --nocapture
```

- `--features "postgres providers"` enables the `postgres` and `providers` feature sets declared in `rustychain/Cargo.toml`. Adjust features as needed.
- `--bench vector_store` selects the bench by name. The final `-- --nocapture` passes `--nocapture` to the benchmark binary so you see console output.

If you want to run all benches for the crate (not just `vector_store`):

```bash
cargo bench -p rustychain --features "postgres providers" -- --nocapture
```

## Notes / Troubleshooting

- Docker must be running and the user must have permission to start containers. On many Linux systems:

```bash
sudo systemctl start docker
sudo usermod -aG docker $USER  # log out/in after this change
```

- If the benchmark panics with:

```
extension "vector" is not available
```

  - That means the Postgres instance doesn't have the `pgvector` extension installed. The bench is written to use a Docker image that already includes `pgvector` (the bench uses a pgvector-ready image such as `ankane/pgvector`), but if you run against a custom Postgres instance you must install the `pgvector` extension there:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

  - If you prefer to use an existing Postgres instead of Docker, set `DATABASE_URL` / `DB_URL` (or modify the bench to read the env var) and ensure the `vector` extension is installed and the DB is reachable.

- If you see an error about `there is no reactor running` or async drop errors, that indicates an async runtime was not available during container cleanup. The bench included in this repo uses a blocking (`testcontainers::clients::Cli`) client to avoid that; ensure you have the updated bench file (uses `ankane/pgvector` and a blocking testcontainers client).

- If you see `Gnuplot not found, using plotters backend` that's harmless — Criterion tries to find gnuplot to render charts. You can ignore the message or install `gnuplot` if you want the gnuplot backend:

```bash
sudo apt install gnuplot  # Debian/Ubuntu
```

- If you run into compilation errors about missing optional crates (e.g. `gemini-rust`, `ollama-rs`, `openai-api-rs`), make sure you enabled the corresponding features declared in `Cargo.toml` (for example `providers` or `openai` / `google` / `ollama`). Example:

```bash
cargo bench -p rustychain --features "postgres providers" --bench vector_store -- --nocapture
```

## Running without Docker (optional)

If you'd rather run benchmarks against a local Postgres instance instead of Docker, ensure:

- Postgres is reachable at the provided connection string and has `pgvector` installed.
- Edit `benches/vector_store.rs` to read a `DATABASE_URL` (or `DB_URL`) environment variable if it's not already set up that way. Example run after editing (or if the bench already honors the env var):

```bash
export DATABASE_URL="postgresql://postgres:postgres@127.0.0.1:5432/postgres"
cargo bench -p rustychain --features "postgres providers" --bench vector_store -- --nocapture
```

## CI considerations

- CI runners must support Docker (or use service containers) for the bench to spawn ephemeral Postgres instances via `testcontainers`.
- Alternative: in CI provision a Postgres service with `pgvector` preinstalled and set `DATABASE_URL` for the benchmark run.

## Useful env vars and flags

- `RUST_BACKTRACE=1` — show backtraces on panic.
- `DATABASE_URL` or `DB_URL` — use an existing Postgres instance if you change the bench to read an env var.

## Contact / next steps

If you want, I can also:

- Modify the `vector_store` bench to prefer a `DATABASE_URL` env var and fall back to testcontainers if not set.
- Add a small shell script `scripts/run_bench_vector_store.sh` that ensures Docker is running and runs the correct `cargo bench` command.

Open an issue or request here if you want either of those additions.
