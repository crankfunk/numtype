#!/usr/bin/env bash
# Kern 06: builds the threads WASM artifact (spike/src/wasm/numtype_core_threads.wasm)
# on the pinned nightly toolchain, per docs/kern-06-threads-spec.md "Toolchain & build".
#
# MUST be invoked with cwd = repo root (Cargo's config-file discovery is
# CWD-based, not --manifest-path-based — see .cargo/config.toml's own
# comment). This script does not `cd`; it trusts the caller (the
# `build:wasm:threads` package.json script runs from the repo root like
# every other pnpm script here).
#
# Deliberately separate from `build:wasm` (stable, .cargo/config.toml-driven):
# this script passes its OWN full RUSTFLAGS (which REPLACES, not merges
# with, the config-file rustflags — so +simd128 is repeated here to keep the
# matmul_blocked.rs compile_error! guard satisfied) and uses its own
# `--target-dir` so the stable build's incremental cache is never touched.
set -euo pipefail

TOOLCHAIN="nightly-2026-07-09"
REPO_ROOT="$(pwd)"
TARGET_DIR="crates/core/target-threads"
OUT_DIR="spike/src/wasm"
OUT_FILE="numtype_core_threads.wasm"

if [ ! -f "crates/core/Cargo.toml" ]; then
  echo "error: build-wasm-threads.sh must be run from the repo root (crates/core/Cargo.toml not found under $REPO_ROOT)." >&2
  exit 1
fi

if ! rustup run "$TOOLCHAIN" rustc --version >/dev/null 2>&1; then
  echo "error: the pinned nightly toolchain '$TOOLCHAIN' is not installed." >&2
  echo "" >&2
  echo "Install it with:" >&2
  echo "  rustup toolchain install $TOOLCHAIN --component rust-src --target wasm32-unknown-unknown" >&2
  exit 1
fi

RUSTFLAGS="-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=1073741824 -C link-arg=--export=__stack_pointer" \
  rustup run "$TOOLCHAIN" cargo build \
    --manifest-path crates/core/Cargo.toml \
    --target wasm32-unknown-unknown \
    --release \
    -Z build-std=std,panic_abort \
    --target-dir "$TARGET_DIR"

mkdir -p "$OUT_DIR"
cp "$TARGET_DIR/wasm32-unknown-unknown/release/numtype_core.wasm" "$OUT_DIR/$OUT_FILE"
echo "built $OUT_DIR/$OUT_FILE"
