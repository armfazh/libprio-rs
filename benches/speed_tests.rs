// SPDX-License-Identifier: MPL-2.0

use criterion::{criterion_group, criterion_main, Criterion};

use prio::field::{random_vector, Field128, FieldElement};

pub fn prng(c: &mut Criterion) {
    let test_sizes = [1, 4, 16, 256, 1024, 4096];
    for size in test_sizes.iter() {
        c.bench_function(&format!("rejection sampling, size={}", *size), |b| {
            b.iter(|| random_vector::<Field128>(*size))
        });
    }

    // TODO Add benchmark for random_vector using pad-then-reduce and Phillipp's method.
}

// TODO Add benchmarks for different prng methods for prio3.

criterion_group!(benches, prng);
criterion_main!(benches);
