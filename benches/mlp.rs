use burn_benches::{bench::BenchSuite, mlp};
use criterion::{criterion_group, criterion_main};

criterion_group!(benches, mlp::MlpBenchSuite::run);
criterion_main!(benches);
