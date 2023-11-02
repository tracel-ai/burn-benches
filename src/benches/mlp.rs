use crate::{
    bench::{run_benchmark, Bench, BenchFunc, BenchSuite},
    device, BenchBackend,
};
use burn::backend::Autodiff;
use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{backend::Backend, Tensor},
};
use criterion::Criterion;

pub struct MlpBenchSuite;

impl BenchSuite for MlpBenchSuite {
    fn name() -> String {
        "mlp".into()
    }

    fn details() -> String {
        let mut details = String::from("Multi layer perceptron (MLP) benchmarks.\n\n");

        configs()
            .into_iter()
            .enumerate()
            .for_each(|(i, config)| details += format!("- {} => `{}`\n", i + 1, config).as_str());

        details
    }

    fn run(c: &mut Criterion) {
        let name = Self::name();
        let name_autodiff = format!("{}-autodiff", Self::name());

        run_benchmark(c, &name, configs(), MlpBench::new());
        run_benchmark(c, &name_autodiff, configs(), MlpBenchAD::new());
    }
}

fn configs() -> Vec<MlpConfig> {
    vec![
        MlpConfig::new(1, 100, 32),
        MlpConfig::new(32, 4, 1024),
        MlpConfig::new(128, 8, 2048),
    ]
}

#[derive(new)]
pub struct MlpBench;
#[derive(new)]
pub struct MlpBenchAD;

impl Bench for MlpBench {
    type Config = MlpConfig;

    fn prepare(&self, config: &Self::Config) -> BenchFunc {
        let device = device();
        let tensor =
            Tensor::<BenchBackend, 2>::ones([config.batch_size, config.d_model]).to_device(&device);
        let mlp = Mlp::new(config).to_device(&device);

        Box::new(move || {
            let tensor = mlp.forward(tensor.clone());
            <BenchBackend as Backend>::sync(&device);
        })
    }
}

impl Bench for MlpBenchAD {
    type Config = MlpConfig;

    fn prepare(&self, config: &Self::Config) -> BenchFunc {
        type ADBackend = Autodiff<BenchBackend>;

        let device = device();
        let tensor =
            Tensor::<ADBackend, 2>::ones([config.batch_size, config.d_model]).to_device(&device);
        let mlp = Mlp::new(config).to_device(&device);

        Box::new(move || {
            let tensor = mlp.forward(tensor.clone());
            let loss = tensor.sum();
            let _grads = loss.backward();
            <BenchBackend as Backend>::sync(&device);
        })
    }
}

#[derive(Config)]
pub struct MlpConfig {
    pub batch_size: usize,
    pub num_layers: usize,
    pub d_model: usize,
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    linears: Vec<nn::Linear<B>>,
    activation: nn::GELU,
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &MlpConfig) -> Self {
        let linears = (0..config.num_layers)
            .map(|_| nn::LinearConfig::new(config.d_model, config.d_model).init())
            .collect();

        Self {
            linears,
            activation: nn::GELU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        for linear in self.linears.iter() {
            x = linear.forward(x);
            x = self.activation.forward(x);
        }

        x
    }
}
