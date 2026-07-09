//! Fill a buffer with a constant value. Doesn't mirror a specific
//! `runtime.ts` function (there `NDArray.zeros`/`.ones` just use
//! `new Float64Array(n)`/`.fill(1)` directly) — it's a general-purpose
//! kernel exposed per the ABI spec, usable by the WASM backend to build its
//! own zeros/ones-equivalent buffers without a copy-in step.

pub fn fill(len: u32, value: f64) -> Vec<f64> {
    vec![value; len as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_basic() {
        assert_eq!(fill(4, 7.0), vec![7.0, 7.0, 7.0, 7.0]);
        assert_eq!(fill(0, 1.0), Vec::<f64>::new());
    }
}
