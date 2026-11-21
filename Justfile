# mesh-dev-plugin tasks

default:
    @just --list

# Verify Cargo.toml and plugin.toml versions match
check-version:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo_ver=$(awk -F'"' '/^version =/{print $2; exit}' Cargo.toml)
    plugin_ver=$(awk -F'"' '/^version =/{print $2; exit}' plugin.toml)
    echo "Cargo.toml=${cargo_ver} plugin.toml=${plugin_ver}"
    test "$cargo_ver" = "$plugin_ver"

# Run formatting, clippy, version check, and full test suite
check: check-version
    cargo fmt --all -- --check
    cargo clippy --locked --all-targets -- -D warnings
    cargo test --locked -- --test-threads 1

# Run full test suite
test:
    cargo test --locked -- --test-threads 1

# Build release binary and package mesh-dev archive matching mesh-llm 0.80.0 install contract
archive target="":
    #!/usr/bin/env bash
    set -euo pipefail
    target="{{target}}"
    if [ -z "$target" ]; then
        target=$(rustc -vV | awk '/^host:/{print $2}')
    fi
    version=$(awk -F'"' '/^version =/{print $2; exit}' plugin.toml)
    cargo build --locked --release --bin mesh-dev
    dist="target/dist"
    rm -rf "$dist"
    stage="${dist}/stage"
    root="${stage}/mesh-dev"
    mkdir -p "$root"
    cp target/release/mesh-dev "$root/mesh-dev"
    cp plugin.toml README.md LICENSE "$root/"
    stable="${dist}/mesh-dev-${target}.tar.gz"
    versioned="${dist}/mesh-dev-${version}-${target}.tar.gz"
    tar czf "$stable" -C "$stage" mesh-dev
    cp "$stable" "$versioned"
    (cd "$dist" && sha256sum "$(basename "$stable")" > "$(basename "$stable").sha256")
    (cd "$dist" && sha256sum "$(basename "$versioned")" > "$(basename "$versioned").sha256")
    rm -rf "$stage"
    echo "Packaged:"
    ls -lh "$dist"
