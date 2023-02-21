use crate::tables::make_tables;
use crate::version_file;
use clap::{Parser, ValueEnum};
use std::fs::File;
use std::io::Write;
use std::process::Command;

static SH_FILENAME: &str = "target/tmp/run_burn.sh";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct BenchesArgs {
    #[arg(short('B'), long, num_args(1..))]
    backends: Vec<Backend>,
    #[arg(short, long, num_args(0..))]
    tags: Vec<String>,
    #[arg(short, long, num_args(0..))]
    commits: Vec<String>,
    #[arg(short, long, num_args(0..))]
    branches: Vec<String>,
    #[arg(short, long, default_value_t = String::from("https://github.com/burn-rs/burn/"))]
    pub repository: String,
}

#[derive(ValueEnum, Debug, Clone)]
pub enum Backend {
    Ndarray,
    NdarrayBlasNetlib,
    NdarrayBlasOpenblas,
    TchCpu,
    TchGpu,
}

#[derive(Clone)]
pub struct BenchParam {
    identifier: String,
    value: String,
    backend_flag: String,
}

impl BenchParam {
    fn bench_filename(&self) -> String {
        format!(
            "target/tmp/{}-{}-{}.json",
            self.backend_flag, self.identifier, self.value
        )
    }
}

pub struct Benches {
    params: Vec<BenchParam>,
    repo: String,
}

impl Drop for Benches {
    fn drop(&mut self) {
        cleanup(&self.params);
    }
}

impl Benches {
    pub fn new(params: Vec<BenchParam>, repo: String) -> Self {
        let params_cloned = params.clone();
        ctrlc::set_handler(move || {
            cleanup(&params_cloned);
        })
        .unwrap();

        Self { params, repo }
    }

    pub fn run(self) {
        prepare(&self.params, &self.repo);

        let mut handle = Command::new("bash").arg(SH_FILENAME).spawn().unwrap();
        handle.wait().unwrap();
    }
}

impl Into<Vec<BenchParam>> for BenchesArgs {
    fn into(self) -> Vec<BenchParam> {
        let mut runs = Vec::new();
        for backend in self.backends {
            let flag = match backend {
                Backend::Ndarray => "ndarray",
                Backend::NdarrayBlasNetlib => "ndarray-blas-netlib",
                Backend::NdarrayBlasOpenblas => "ndarray-blas-openblas",
                Backend::TchCpu => "tch-cpu",
                Backend::TchGpu => "tch-gpu",
            };
            for tag in self.tags.iter() {
                runs.push(BenchParam {
                    identifier: "tag".into(),
                    value: tag.into(),
                    backend_flag: flag.into(),
                });
            }
            for commit in self.commits.iter() {
                runs.push(BenchParam {
                    identifier: "rev".into(),
                    value: commit.into(),
                    backend_flag: flag.into(),
                });
            }
            for branch in self.branches.iter() {
                runs.push(BenchParam {
                    identifier: "branch".into(),
                    value: branch.into(),
                    backend_flag: flag.into(),
                });
            }
        }

        runs
    }
}

fn write_bash_file(runs: &[BenchParam], filename: &str, repo: &str) {
    let mut content = String::new();

    for run in runs.iter() {
        content += build_bash(run, repo).as_str();
    }
    content += "cat ";
    for run in runs.iter() {
        content += run.bench_filename().as_str();
        content += " ";
    }

    content += "| criterion-table >  BENCHMARKS.md";

    if let Some(_) = Command::new("pandoc").arg("--help").output().ok() {
        content += "\npandoc -f markdown BENCHMARKS.md > benchmarks.html";
    }

    let mut file = File::create(filename).unwrap();
    write!(file, "{}", content).unwrap();
}

fn prepare(runs: &[BenchParam], repo: &str) {
    write_bash_file(&runs, SH_FILENAME, repo);
    make_tables();

    Command::new("cp")
        .args(["Cargo.toml", "Cargo-tmp.toml"])
        .output()
        .unwrap();
}

fn cleanup(params: &[BenchParam]) {
    Command::new("mv")
        .args(["Cargo-tmp.toml", "Cargo.toml"])
        .output()
        .unwrap();

    for run in params.iter() {
        std::fs::remove_file(run.bench_filename()).ok();
    }

    std::fs::remove_file(SH_FILENAME).ok();
    std::fs::remove_file("tables.toml").ok();
}

fn build_bash(run: &BenchParam, repo: &str) -> String {
    let ops = format!("--no-default-features --features {}", run.backend_flag);

    let identifier = &run.identifier;
    let value = &run.value;

    let mut output = String::new();
    output += format!("echo {} > {}\n", run.value, version_file()).as_str();
    output += format!("cargo add burn --git {repo} --{identifier} {value}\n").as_str();
    output += format!("cargo add burn-tch --git {repo} --{identifier} {value}\n").as_str();
    output += format!("cargo add burn-ndarray --git {repo} --{identifier} {value}\n").as_str();
    output += format!(
        "cargo criterion {ops} --message-format=json > {}\n",
        run.bench_filename()
    )
    .as_str();
    output
}
