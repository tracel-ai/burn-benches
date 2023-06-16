use crate::{
    bench::{run_benchmark, Bench, BenchFunc, BenchSuite},
    device, BenchBackend,
};
use burn::{
    config::Config,
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig, Conv2dPaddingConfig},
    tensor::{backend::Backend, Distribution, Tensor},
};
use burn_autodiff::ADBackendDecorator;
use criterion::Criterion;

pub struct Conv2dBenchSuite;

impl BenchSuite for Conv2dBenchSuite {
    fn name() -> String {
        "Conv2d".into()
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
    convs: Vec<Conv2d<B>>,
}

impl<B: Backend> Conv2dBlock<B> {
    pub fn new(config: &Conv2dBenchConfig) -> Self {
        let convs = (0..config.num_layers)
            .map(|_| config.conv2d.init())
            .collect();

        Self { convs }
    }

    pub fn forward(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = tensor;

        for conv in self.convs.iter() {
            x = conv.forward(x);
        }

        x
    }
}

fn configs() -> Vec<Conv2dBenchConfig> {
    vec![
        Conv2dBenchConfig::new(
            4,
            32,
            32,
            2,
            Conv2dConfig::new([1, 1], [3, 3]).with_padding(Conv2dPaddingConfig::Same),
        ),
        Conv2dBenchConfig::new(
            4,
            64,
            64,
            2,
            Conv2dConfig::new([1, 1], [3, 3]).with_padding(Conv2dPaddingConfig::Same),
        ),
    ]
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
        let module = Conv2dBlock::new(config).to_device(&device);

        Box::new(move || {
            let _tensor = module.forward(tensor.clone());
        })
    }
}

impl Bench for Conv2dBenchAD {
    type Config = Conv2dBenchConfig;

    fn prepare(&self, config: &Self::Config) -> BenchFunc {
        type Backend = ADBackendDecorator<BenchBackend>;

        let device = device();
        let tensor = Tensor::<Backend, 4>::random(
            [
                config.batch_size,
                config.conv2d.channels[0],
                config.height,
                config.width,
            ],
            Distribution::Standard,
        )
        .to_device(&device);
        let module = Conv2dBlock::new(config).to_device(&device);

        Box::new(move || {
            let tensor = module.forward(tensor.clone());
            let _grads = tensor.backward();
        })
    }
}
