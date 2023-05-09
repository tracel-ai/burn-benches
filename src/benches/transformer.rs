use crate::{
    bench::{run_benchmark, Bench, BenchFunc, BenchSuite},
    device, BenchBackend,
};
use burn::{
    config::Config,
    module::Module,
    nn::transformer::{TransformerEncoderConfig, TransformerEncoderInput},
    tensor::Tensor,
};
use burn_autodiff::ADBackendDecorator;
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
        let name = Self::name();
        let name_autodiff = format!("{}-autodiff", Self::name());

        run_benchmark(c, &name, configs(), TansformerBench::new());
        run_benchmark(c, &name_autodiff, configs(), TansformerBenchAD::new());
    }
}

fn configs() -> Vec<TransformerConfig> {
    vec![
        TransformerConfig::new(4, 128, TransformerEncoderConfig::new(64, 256, 4, 4)),
        TransformerConfig::new(4, 128, TransformerEncoderConfig::new(256, 1024, 8, 4)),
    ]
}

#[derive(new)]
pub struct TansformerBench;
#[derive(new)]
pub struct TansformerBenchAD;

impl Bench for TansformerBench {
    type Config = TransformerConfig;

    fn prepare(&self, config: &Self::Config) -> BenchFunc {
        let device = device();
        let tensor = Tensor::<BenchBackend, 3>::ones([
            config.batch_size,
            config.seq_length,
            config.encoder.d_model,
        ])
        .to_device(&device);
        let transformer = config.encoder.init().to_device(&device);

        Box::new(move || {
            let input = TransformerEncoderInput::new(tensor.clone());
            let _tensor = transformer.forward(input);
        })
    }
}

impl Bench for TansformerBenchAD {
    type Config = TransformerConfig;

    fn prepare(&self, config: &Self::Config) -> BenchFunc {
        type Backend = ADBackendDecorator<BenchBackend>;

        let device = device();
        let tensor = Tensor::<Backend, 3>::ones([
            config.batch_size,
            config.seq_length,
            config.encoder.d_model,
        ])
        .to_device(&device);
        let transformer = config.encoder.init().to_device(&device);

        Box::new(move || {
            let input = TransformerEncoderInput::new(tensor.clone());
            let tensor = transformer.forward(input);
            let _grads = tensor.backward();
        })
    }
}
#[derive(Config)]
pub struct TransformerConfig {
    pub batch_size: usize,
    pub seq_length: usize,
    pub encoder: TransformerEncoderConfig,
}
