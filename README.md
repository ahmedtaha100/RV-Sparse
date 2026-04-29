# RV-Sparse Coding Challenge

Repository: https://github.com/ahmedtaha100/R

## Overview

This project implements `sparse_multiply` in `challenge.c`.

The function scans a dense row-major matrix, writes its non-zero entries into caller-provided CSR buffers, and computes the matrix-vector product `y = A * x`.

## Build and Run

```sh
gcc -o run challenge.c -lm
./run
```

Pass an optional integer seed to reproduce a specific randomized test run:

```sh
./run 123456789
```

## Implementation Notes

`sparse_multiply` performs no dynamic memory allocation. It uses only the buffers supplied by the caller:

- `values`: non-zero matrix values in row-major order, with capacity for `rows * cols` entries.
- `col_indices`: column index for each non-zero value, with capacity for `rows * cols` entries.
- `row_ptrs`: CSR row offsets with `row_ptrs[0] = 0` and `row_ptrs[rows] = out_nnz`.
- `y`: matrix-vector product.

The input vector `x` and output vector `y` must be separate buffers.

`sparse_multiply_checked` is available for callers that want explicit capacity validation while preserving the challenge-required `sparse_multiply` signature.
