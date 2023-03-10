use crate::{
    bench::{run_benchmark, Bench, BenchFunc, BenchSuite},
    device, BenchBackend,
};
use burn::{
    config::Config,
    module::{Module, Param},
    nn::conv::{Conv2d, Conv2dConfig},
    tensor::{backend::Backend, Distribution, Tensor},
};
use burn_autodiff::ADBackendDecorator;
use criterion::Criterion;

pub struct TransformerBenchSuite;

impl BenchSuite for TransformerBenchSuite {
    fn name() -> String {
        "transformer".into()
    }

    fn details() -> String {
        let mut details = String::from("Conv2d benchmarks.\n\n");

        configs()
            .into_iter()
            .enumerate()
            .for_each(|(i, config)| details += format!("- {} => `{}`\n", i + 1, config).as_str());

        details
    }

    fn run(c: &mut Criterion) {
        let name = Self::name();
        let name_autodiff = format!("{}-autodiff", Self::name());

        run_benchmark(c, &name, configs(), Conv2dBench::new());
        run_benchmark(c, &name_autodiff, configs(), Conv2dBenchAD::new());
    }
}

#[derive(Config)]
pub struct Conv2dBenchConfig {
    pub batch_size: usize,
    pub height: usize,
    pub width: usize,
    pub num_layers: usize,
    pub conv2d: Conv2dConfig,
}

#[derive(Module, Debug)]
pub struct Conv2dBlock<B: Backend> {
    convs: Param<Vec<Conv2d<B>>>,
}

impl<B: Backend> Conv2dBlock<B> {
    pub fn new(config: &Conv2dBenchConfig) -> Self {
        let convs = (0..config.num_layers)
            .map(|_| Conv2d::new(&config.conv2d))
            .collect::<Vec<_>>();

        Self {
            convs: Param::from(convs),
        }
    }
}

fn configs() -> Vec<Conv2dConfig> {
    vec![]
}

#[derive(new)]
pub struct Conv2dBench;
#[derive(new)]
pub struct Conv2dBenchAD;

impl Bench for Conv2dBench {
    type Config = Conv2dBenchConfig;

    fn prepare(&self, config: &Self::Config) -> BenchFunc {
        let device = device();
        let tensor = Tensor::<BenchBackend, 4>::random(
            [
                config.batch_size,
                config.conv2d.channels[0],
                config.height,
                config.width,
            ],
            Distribution::Standard,
        )
        .to_device(&device);
        let module = Conv2dBlock::new(&config).to_device(&device);

        Box::new(move || {
            let _tensor = module.forward(input.clone());
        })
    }
}

impl Bench for Conv2dBenchAD {
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
        let mut transformer = TransformerEncoder::new(&config.encoder);
        transformer.to_device(&device);

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
