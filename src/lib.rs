#[macro_use]
extern crate derive_new;

pub mod bench;
pub mod cli;
pub mod tables;

mod benches;
pub use benches::*;

#[cfg(feature = "tch-cpu")]
pub type BenchBackend = burn_tch::TchBackend<f32>;
#[cfg(feature = "tch-gpu")]
pub type BenchBackend = burn_tch::TchBackend<f32>;

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-no-std"
))]
pub type BenchBackend = burn_ndarray::NdArrayBackend<f32>;

pub type BenchDevice = <BenchBackend as burn::tensor::backend::Backend>::Device;

pub fn bench_id() -> String {
    format!("{}:{}", flags(), version())
}

pub fn flags() -> String {
    #[cfg(feature = "tch-cpu")]
    return "tch-cpu".into();

    #[cfg(feature = "tch-gpu")]
    return "tch-gpu".into();

    #[cfg(feature = "ndarray")]
    return "ndarray".into();

    #[cfg(feature = "ndarray-blas-netlib")]
    return "ndarray-netlib".into();

    #[cfg(feature = "ndarray-blas-openblas")]
    return "ndarray-openblas".into();

    #[cfg(feature = "ndarray-no-std")]
    return "ndarray-no-std".into();
}

pub fn device() -> BenchDevice {
    #[cfg(feature = "tch-gpu")]
    return burn_tch::TchDevice::Cuda(0);

    #[cfg(feature = "tch-cpu")]
    return burn_tch::TchDevice::Cpu;

    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-no-std"
    ))]
    return burn_ndarray::NdArrayDevice::Cpu;
}

pub fn version_file() -> String {
    String::from("target/tmp/version_burn")
}

pub fn version() -> String {
    let version = std::fs::read_to_string(version_file()).unwrap();
    let version = version.trim();
    version.to_string()
}
