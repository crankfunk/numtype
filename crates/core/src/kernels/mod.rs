pub mod add;
pub mod fill;
pub mod materialize;
pub mod matmul;
pub mod matmul_blocked;
pub mod sum;
pub mod transpose;
// Kern 07: elementwise sub/mul/div + dot/norm_sq vector reductions.
// Appended after all pre-existing module declarations (freeze discipline).
pub mod elementwise;
pub mod vector;
