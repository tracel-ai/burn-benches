use burn_benches::{bench::BenchSuite, conv2d};
use criterion::{criterion_group, criterion_main};

criterion_group!(benches, conv2d::Conv2dBenchSuite::run);
criterion_main!(benches);
