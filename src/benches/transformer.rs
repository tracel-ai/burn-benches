use crate::{
    bench::{run_benchmark, Bench, BenchFunc, BenchSuite},
    device, BenchBackend,
};
use burn::{
    config::Config,
    module::Module,
    nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    tensor::Tensor,
};
use criterion::Criterion;

pub struct TransformerBenchSuite;

impl BenchSuite for TransformerBenchSuite {
    fn name() -> String {
        "transformer".into()
    }

    fn details() -> String {
        let mut details = String::from("Transformer encoder benchmarks.\n\n");

        configs()
            .into_iter()
            .enumerate()
            .for_each(|(i, config)| details += format!("- {} => `{}`\n", i + 1, config).as_str());

        details
    }

    fn run(c: &mut Criterion) {
        run_benchmark(c, &Self::name(), configs(), MhaBench::new());
    }
}

fn configs() -> Vec<TransformerConfig> {
    vec![
        TransformerConfig::new(4, 128, TransformerEncoderConfig::new(64, 256, 4, 4)),
        TransformerConfig::new(4, 128, TransformerEncoderConfig::new(128, 512, 8, 4)),
        TransformerConfig::new(4, 128, TransformerEncoderConfig::new(256, 1024, 8, 4)),
        TransformerConfig::new(4, 128, TransformerEncoderConfig::new(512, 2048, 16, 4)),
    ]
}

#[derive(new)]
pub struct MhaBench;

impl Bench for MhaBench {
    type Config = TransformerConfig;

    fn prepare(&self, config: &Self::Config) -> BenchFunc {
        let device = device();
        let tensor = Tensor::<BenchBackend, 3>::ones([
            config.batch_size,
            config.seq_length,
            config.encoder.d_model,
        ])
        .to_device(&device);
        let mut transformer = TransformerEncoder::new(&config.encoder);
        transformer.to_device(&device);

        return Box::new(move || {
            let input = TransformerEncoderInput::new(tensor.clone());
            let _tensor = transformer.forward(input);
        });
    }
}

#[derive(Config)]
pub struct TransformerConfig {
    pub batch_size: usize,
    pub seq_length: usize,
    pub encoder: TransformerEncoderConfig,
}
