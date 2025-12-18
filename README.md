# RustyChain

A high-performance, type-safe Rust library for building LLM-powered applications, chains, and agents.

## Installation

Add RustyChain to your project using Cargo:

```bash
cargo add rustychain
```

Add to your `Cargo.toml`:

```toml
[dependencies]
rustychain = { version = "0.0.1", features = ["essential"] }
```

or get the latest version from GitHub:

```toml
[dependencies]
rustychain = { git = "https://github.com/JeanVydes/rustychain", branch = "main", features = ["essential"] }
``` 

## Start

```rust
use rustychain::{LLM, LLMProvider};

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let llm = LLM::builder()
        .set_authorization("my-api-key")
        .set_model("gemini-2.5-flash")
        .set_provider(LLMProvider::Google)
        .set_system_prompt("You are a helpful assistant.")
        .build()?;

    let response = llm.inference("Hello, how are you?").generate().await?;
    println!("LLM response: {}", response);

    Ok(())
}
```

## Benchmarks

See the [BENCHMARKS.md](BENCHMARKS.md) file for detailed performance benchmarks.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## Documentation

Project documentation is in the `docs/` folder. There are two targeted doc sets:

- `docs/contributors/` — Developer-focused guides (building, testing, contributing).
- `docs/wiki/` — User-facing guides and how-tos suitable for a GitHub Wiki or publishing to a website.

Open `docs/README.md` for an index of available guides.

---

Built with ❤️
