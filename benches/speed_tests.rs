// SPDX-License-Identifier: MPL-2.0

use criterion::{criterion_group, criterion_main, Criterion};

use prio::field::{random_vector, random_vector2, random_vector3, Field128};

// Benchmark for random_vector using rejection sampling
pub fn prng(c: &mut Criterion) {
    let test_sizes = [1, 4, 16, 256, 1024, 4096];
    for size in test_sizes.iter() {
        c.bench_function(&format!("rejection sampling, size={}", *size), |b| {
            b.iter(|| random_vector::<Field128>(*size))
        });
    }
}

// Benchmark for random_vector using pad-then-reduce and
pub fn prng2(c: &mut Criterion) {
    let test_sizes = [16, 256, 1024, 4096];
    for size in test_sizes.iter() {
        c.bench_function(&format!("pad-then-reduce, size={}", *size), |b| {
            b.iter(|| random_vector2::<Field128>(*size))
        });
    }
}

// Benchmark for random_vector using hash_to_curve and
pub fn prng3(c: &mut Criterion) {
    let test_sizes = [16, 256, 1024, 4096];
    for size in test_sizes.iter() {
        c.bench_function(&format!("hash_to_curve, size={}", *size), |b| {
            b.iter(|| random_vector3::<Field128>(*size))
        });
    }
}

// TODO Phillipp's method.

// TODO Add benchmarks for different prng methods for prio3.
criterion_group!(benches, prng, prng2, prng3);
criterion_main!(benches);
