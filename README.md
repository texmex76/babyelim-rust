# babyelim-rust

This is the Rust template for the SAT variable elimination exercise.

I had no time to implement a solution for myself yet, so I am unsure if there are any bugs.

If you have any suggestions or find some bugs, pull requests are welcome!

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
