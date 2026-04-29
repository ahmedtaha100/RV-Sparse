#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

void sparse_multiply(
    int rows,
    int cols,
    const double* A,
    const double* x,
    int* out_nnz,
    double* values,
    int* col_indices,
    int* row_ptrs,
    double* y
);

static int fail_sparse_multiply(int* out_nnz) {
    if (out_nnz) {
        *out_nnz = -1;
    }

    return -1;
}

int sparse_multiply_checked(
    int rows,
    int cols,
    const double* A,
    const double* x,
    size_t values_cap,
    size_t col_indices_cap,
    size_t row_ptrs_cap,
    size_t y_cap,
    int* out_nnz,
    double* values,
    int* col_indices,
    int* row_ptrs,
    double* y
) {
    size_t row_count = 0;
    size_t required_nnz = 0;
    size_t nnz = 0;

    if (rows < 0 || cols < 0 || !A || !x || !out_nnz || !values || !col_indices || !row_ptrs || !y) {
        return fail_sparse_multiply(out_nnz);
    }

    if (x == y) {
        return fail_sparse_multiply(out_nnz);
    }

    row_count = (size_t)rows;

    if (row_ptrs_cap < row_count + 1U || y_cap < row_count) {
        return fail_sparse_multiply(out_nnz);
    }

    for (int i = 0; i < rows; ++i) {
        const double* row = A + (size_t)i * (size_t)cols;

        for (int j = 0; j < cols; ++j) {
            if (row[j] != 0.0) {
                if (required_nnz >= (size_t)INT_MAX) {
                    return fail_sparse_multiply(out_nnz);
                }

                ++required_nnz;
            }
        }
    }

    if (values_cap < required_nnz || col_indices_cap < required_nnz) {
        return fail_sparse_multiply(out_nnz);
    }

    row_ptrs[0] = 0;

    for (int i = 0; i < rows; ++i) {
        const double* row = A + (size_t)i * (size_t)cols;

        for (int j = 0; j < cols; ++j) {
            double value = row[j];

            if (value != 0.0) {
                values[nnz] = value;
                col_indices[nnz] = j;
                ++nnz;
            }
        }

        row_ptrs[i + 1] = (int)nnz;
    }

    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;

        for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
            sum += values[k] * x[col_indices[k]];
        }

        y[i] = sum;
    }

    *out_nnz = (int)nnz;
    return 0;
}

void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    size_t row_count = 0;
    size_t col_count = 0;
    size_t max_possible_nnz = 0;

    if (!out_nnz) {
        return;
    }

    if (rows < 0 || cols < 0) {
        *out_nnz = -1;
        return;
    }

    row_count = (size_t)rows;
    col_count = (size_t)cols;
    if (col_count != 0U && row_count > SIZE_MAX / col_count) {
        max_possible_nnz = SIZE_MAX;
    } else {
        max_possible_nnz = row_count * col_count;
    }

    (void)sparse_multiply_checked(
        rows,
        cols,
        A,
        x,
        max_possible_nnz,
        max_possible_nnz,
        row_count + 1U,
        row_count,
        out_nnz,
        values,
        col_indices,
        row_ptrs,
        y
    );
}

static void free_buffers(
    double* A,
    double* values,
    int* col_indices,
    int* row_ptrs,
    double* x,
    double* y_user,
    double* y_ref
) {
    free(A);
    free(values);
    free(col_indices);
    free(row_ptrs);
    free(x);
    free(y_user);
    free(y_ref);
}

static int report_csr_error(int iter, const char* message, int row, int index) {
    fprintf(stderr, "Iter %d CSR error: %s (row=%d, index=%d)\n", iter, message, row, index);
    return 0;
}

static int validate_csr(
    int iter,
    int rows,
    int cols,
    const double* A,
    int out_nnz,
    const double* values,
    const int* col_indices,
    const int* row_ptrs
) {
    size_t mat_sz = (size_t)rows * (size_t)cols;

    if (out_nnz < 0) {
        return report_csr_error(iter, "negative non-zero count", -1, out_nnz);
    }

    if ((size_t)out_nnz > mat_sz) {
        return report_csr_error(iter, "non-zero count exceeds matrix size", -1, out_nnz);
    }

    if (row_ptrs[0] != 0) {
        return report_csr_error(iter, "row_ptrs[0] is not zero", 0, row_ptrs[0]);
    }

    if (row_ptrs[rows] != out_nnz) {
        return report_csr_error(iter, "last row pointer does not match non-zero count", rows, row_ptrs[rows]);
    }

    for (int r = 0; r < rows; ++r) {
        int start = row_ptrs[r];
        int end = row_ptrs[r + 1];

        if (start < 0 || end < 0) {
            return report_csr_error(iter, "negative row pointer", r, start < 0 ? start : end);
        }

        if (start > end) {
            return report_csr_error(iter, "row pointers are not non-decreasing", r, start);
        }

        if (end > out_nnz) {
            return report_csr_error(iter, "row pointer exceeds non-zero count", r, end);
        }
    }

    for (int k = 0; k < out_nnz; ++k) {
        if (col_indices[k] < 0 || col_indices[k] >= cols) {
            return report_csr_error(iter, "column index out of range", -1, k);
        }

        if (!isfinite(values[k])) {
            return report_csr_error(iter, "non-finite CSR value", -1, k);
        }
    }

    for (int r = 0; r < rows; ++r) {
        int cursor = row_ptrs[r];
        int end = row_ptrs[r + 1];

        for (int c = 0; c < cols; ++c) {
            double value = A[(size_t)r * (size_t)cols + (size_t)c];

            if (value != 0.0) {
                if (cursor >= end) {
                    return report_csr_error(iter, "missing CSR entry", r, c);
                }

                if (col_indices[cursor] != c) {
                    return report_csr_error(iter, "CSR column order or content mismatch", r, cursor);
                }

                if (values[cursor] != value) {
                    return report_csr_error(iter, "CSR value mismatch", r, cursor);
                }

                ++cursor;
            }
        }

        if (cursor != end) {
            return report_csr_error(iter, "extra CSR entries in row", r, cursor);
        }
    }

    return 1;
}

static int parse_seed(int argc, char** argv, unsigned int* seed) {
    char* end = NULL;
    unsigned long parsed = 0;

    if (argc <= 1) {
        *seed = 123456789U;
        return 1;
    }

    errno = 0;
    parsed = strtoul(argv[1], &end, 10);
    if (errno != 0 || end == argv[1] || *end != '\0' || parsed > UINT_MAX) {
        fprintf(stderr, "Invalid seed: %s\n", argv[1]);
        return 0;
    }

    *seed = (unsigned int)parsed;
    return 1;
}

int main(int argc, char** argv) {
    unsigned int seed = 0;

    if (!parse_seed(argc, argv, &seed)) {
        return 1;
    }

    srand(seed);
    printf("Seed: %u\n", seed);

    const int num_iterations = 100;
    int passed_count = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        int rows = rand() % 41 + 5;
        int cols = rand() % 41 + 5;
        double density = 0.05 + (rand() / (double) RAND_MAX) * 0.35;

        size_t mat_sz = (size_t)rows * (size_t)cols;

        double* A = NULL;
        double* values = NULL;
        int* col_indices = NULL;
        int* row_ptrs = NULL;
        double* x = NULL;
        double* y_user = NULL;
        double* y_ref = NULL;
        int out_nnz = 0;

        A = calloc(mat_sz, sizeof(double));
        if (!A) {
            fprintf(stderr, "Allocation failed on iter %d\n", iter);
            free_buffers(A, values, col_indices, row_ptrs, x, y_user, y_ref);
            return 1;
        }

        values = malloc(mat_sz * sizeof(double));
        if (!values) {
            fprintf(stderr, "Allocation failed on iter %d\n", iter);
            free_buffers(A, values, col_indices, row_ptrs, x, y_user, y_ref);
            return 1;
        }

        col_indices = malloc(mat_sz * sizeof(int));
        if (!col_indices) {
            fprintf(stderr, "Allocation failed on iter %d\n", iter);
            free_buffers(A, values, col_indices, row_ptrs, x, y_user, y_ref);
            return 1;
        }

        row_ptrs = malloc(((size_t)rows + 1U) * sizeof(int));
        if (!row_ptrs) {
            fprintf(stderr, "Allocation failed on iter %d\n", iter);
            free_buffers(A, values, col_indices, row_ptrs, x, y_user, y_ref);
            return 1;
        }

        x = malloc((size_t)cols * sizeof(double));
        if (!x) {
            fprintf(stderr, "Allocation failed on iter %d\n", iter);
            free_buffers(A, values, col_indices, row_ptrs, x, y_user, y_ref);
            return 1;
        }

        y_user = malloc((size_t)rows * sizeof(double));
        if (!y_user) {
            fprintf(stderr, "Allocation failed on iter %d\n", iter);
            free_buffers(A, values, col_indices, row_ptrs, x, y_user, y_ref);
            return 1;
        }

        y_ref = calloc((size_t)rows, sizeof(double));
        if (!y_ref) {
            fprintf(stderr, "Allocation failed on iter %d\n", iter);
            free_buffers(A, values, col_indices, row_ptrs, x, y_user, y_ref);
            return 1;
        }

        for (size_t i = 0; i < mat_sz; ++i) {
            if (((double) rand() / RAND_MAX) < density) {
                A[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
            }
        }

        for (int i = 0; i < cols; ++i) {
            x[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
        }

        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols; ++j) {
                sum += A[(size_t)i * (size_t)cols + (size_t)j] * x[j];
            }
            y_ref[i] = sum;
        }

        sparse_multiply(rows, cols, A, x, &out_nnz, values, col_indices, row_ptrs, y_user);

        double max_err = 0.0;
        int csr_valid = validate_csr(iter, rows, cols, A, out_nnz, values, col_indices, row_ptrs);
        int numeric_pass = 1;

        if (csr_valid) {
            for (int i = 0; i < rows; ++i) {
                double diff = fabs(y_user[i] - y_ref[i]);
                double tol = 1e-7 + 1e-7 * fabs(y_ref[i]);
                if (diff > tol) {
                    max_err = fmax(max_err, diff);
                    numeric_pass = 0;
                }
            }
        }

        if (csr_valid && numeric_pass) {
            passed_count++;
        }

        if (!csr_valid) {
            printf(
                "Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: FAIL (CSR validation failed)\n",
                iter, rows, cols, density, out_nnz
            );
        } else {
            printf(
                "Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: %s (Max error: %.2e)\n",
                iter, rows, cols, density, out_nnz, numeric_pass ? "PASS" : "FAIL", max_err
            );
        }

        free_buffers(A, values, col_indices, row_ptrs, x, y_user, y_ref);
    }

    printf(
        "\n%s (%d/%d iterations passed)\n",
        passed_count == num_iterations ? "All tests passed!" : "Some tests failed.",
        passed_count, num_iterations
    );

    return passed_count == num_iterations ? 0 : 1;
}
