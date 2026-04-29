# RV-Sparse Coding Challenge

Repository: https://github.com/ahmedtaha100/R

## Overview

This project implements `sparse_multiply` in `challenge.c`.

The function scans a dense row-major matrix, writes its non-zero entries into caller-provided CSR buffers, and computes the matrix-vector product `y = A * x`.

## Build and Run

```sh
gcc -lm -o run challenge.c
./run
```

## Implementation Notes

`sparse_multiply` performs no dynamic memory allocation. It uses only the buffers supplied by the caller:

- `values` stores non-zero matrix values in row-major order.
- `col_indices` stores the column index for each non-zero value.
- `row_ptrs` stores CSR row offsets with `row_ptrs[0] = 0` and `row_ptrs[rows] = out_nnz`.
- `y` stores the matrix-vector product.
