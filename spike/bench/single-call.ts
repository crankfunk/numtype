/**
 * Minimal consumer file: exactly one import + one `add` + one `matmul`
 * call site. Compiled in isolation (see spike/bench/tsconfig.json) via
 * `pnpm check:diag:bench` to measure the marginal `--extendedDiagnostics`
 * cost (Instantiations / Check time / Memory) of using the type layer at
 * all, separate from the full test suite's numbers.
 */
import { NDArray } from "../src/ndarray.ts";

const a = NDArray.zeros([2, 3]);
const b = NDArray.zeros([3]);
const added = a.add(b);

const m1 = NDArray.zeros([2, 3]);
const m2 = NDArray.zeros([3, 4]);
const multiplied = m1.matmul(m2);

console.log(added.shape, multiplied.shape);
