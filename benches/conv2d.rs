use burn_benches::{bench::BenchSuite, transformer};
use criterion::{criterion_group, criterion_main};

criterion_group!(benches, transformer::TransformerBenchSuite::run);
criterion_main!(benches);
