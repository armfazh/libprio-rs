// SPDX-License-Identifier: MPL-2.0

use criterion::{criterion_group, criterion_main, Criterion};

use prio::field::{random_vector, random_vector_pad_then_reduce, Field128};

const TEST_SIZES: [usize; 3] = [256, 1024, 4096];

// Benchmark for generating field elements using rejection sampling method.
pub fn prng_rejection_sampling(c: &mut Criterion) {
    for size in TEST_SIZES.iter() {
        c.bench_function(&format!("rejection sampling, size={}", *size), |b| {
            b.iter(|| random_vector::<Field128>(*size))
        });
    }
}

// Benchmark for generating field elements using pad and reduce method.
pub fn prng_pad_reduce(c: &mut Criterion) {
    for size in TEST_SIZES.iter() {
        c.bench_function(&format!("pad_and_reduce, size={}", *size), |b| {
            b.iter(|| random_vector_pad_then_reduce::<Field128>(*size))
        });
    }
}

// TODO Phillipp's method.

criterion_group!(benches, prng_rejection_sampling, prng_pad_reduce,);
criterion_main!(benches);
