use std::time::Duration;

use criterion::{black_box, BenchmarkId, Criterion};

use crate::bench_id;

pub trait BenchSuite {
    fn name() -> String;
    fn details() -> String;
    fn run(c: &mut Criterion);
}

pub trait Bench {
    type Config: std::fmt::Display;

    fn prepare(&self, config: &Self::Config) -> BenchFunc;
}

pub type BenchBoxed<C> = Box<dyn Bench<Config = C>>;
pub type BenchFunc = Box<dyn FnMut()>;

pub fn run_benchmark<C, B>(c: &mut Criterion, name: &str, configs: Vec<C>, bench: B)
where
    C: std::fmt::Display,
    B: Bench<Config = C>,
{
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(250));

    configs.iter().enumerate().for_each(|(i, config)| {
        let mut func = bench.prepare(config);

        group.bench_with_input(BenchmarkId::new(bench_id(), i + 1), &(), |b, _i| {
            b.iter(|| {
                func();
                black_box(())
            })
        });
    });
}
