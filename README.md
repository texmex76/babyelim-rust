# babyelim-rust

This is a Rust solution for the SAT variable elimination exercise.

# Running

```
cargo run -- [OPTIONS] [CNF PATH] [OUT PATH]
```

# Testing

The tests hang if you do not provide a solution for the exercise.

```
cargo test
```

# Logging

```
cargo run --features "logging" -- [OPTIONS] [CNF PATH] [OUT PATH] [PROOF PATH]
```
