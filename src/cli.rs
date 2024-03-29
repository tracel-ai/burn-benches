use crate::tables::make_tables;
use crate::version_file;
use clap::{Parser, ValueEnum};
use std::fs::File;
use std::io::Write;
use std::process::Command;

static SH_FILENAME: &str = "target/tmp/run_burn.sh";
static OUTPUT_DIR: &str = "target/burn_benches";
static MD_FILENAME: &str = "target/burn_benches/BENCHMARKS.md";
static HTML_FILENAME: &str = "target/burn_benches/benchmarks.html";

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
    #[arg(short, long, num_args(0..))]
    paths: Vec<String>,
    #[arg(short('N'), long)]
    bench: Bench,
    #[arg(short, long, default_value_t = String::from("https://github.com/burn-rs/burn/"))]
    pub repository: String,
}

#[derive(ValueEnum, Debug, Clone)]
pub enum Backend {
    Ndarray,
    NdarrayNetlib,
    NdarrayOpenblas,
    NdarrayNoStd,
    TchCpu,
    TchGpu,
    Wgpu,
}

#[derive(ValueEnum, Debug, Clone)]
pub enum Bench {
    Transformer,
    MLP,
    Conv2d,
    All,
}

#[derive(Clone)]
pub enum BenchParam {
    Path(BenchSettings),
    Git(BenchSettings),
}

impl BenchSettings {
    fn bench_filename(&self) -> String {
        let value = self.value.replace('/', "-");

        format!(
            "{}/{}-{}-{}.json",
            OUTPUT_DIR, self.backend_flag, self.identifier, value
        )
    }
}
impl BenchParam {
    pub fn settings(&self) -> &BenchSettings {
        match self {
            BenchParam::Path(val) => val,
            BenchParam::Git(val) => val,
        }
    }
}

#[derive(Clone)]
pub struct BenchSettings {
    identifier: String,
    value: String,
    backend_flag: String,
    bench: String,
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

        let mut handle = Command::new("sh").arg(SH_FILENAME).spawn().unwrap();
        handle.wait().unwrap();
    }
}

impl Into<Vec<BenchParam>> for BenchesArgs {
    fn into(self) -> Vec<BenchParam> {
        let mut bench = String::new();

        match self.bench {
            Bench::Transformer => bench += "transformer",
            Bench::MLP => bench += "mlp",
            Bench::Conv2d => bench += "conv2d",
            Bench::All => {}
        }

        if !bench.is_empty() {
            bench = format!("--bench {bench}");
        }

        let mut runs = Vec::new();
        for backend in self.backends {
            let flag = match backend {
                Backend::Ndarray => "ndarray",
                Backend::NdarrayNetlib => "ndarray-blas-netlib",
                Backend::NdarrayOpenblas => "ndarray-blas-openblas",
                Backend::NdarrayNoStd => "ndarray-no-std",
                Backend::TchCpu => "tch-cpu",
                Backend::TchGpu => "tch-gpu",
                Backend::Wgpu => "wgpu",
            };

            for tag in self.tags.iter() {
                runs.push(BenchParam::Git(BenchSettings {
                    identifier: "tag".into(),
                    value: tag.into(),
                    bench: bench.clone(),
                    backend_flag: flag.into(),
                }));
            }
            for commit in self.commits.iter() {
                runs.push(BenchParam::Git(BenchSettings {
                    identifier: "rev".into(),
                    value: commit.into(),
                    bench: bench.clone(),
                    backend_flag: flag.into(),
                }));
            }
            for branch in self.branches.iter() {
                runs.push(BenchParam::Git(BenchSettings {
                    identifier: "branch".into(),
                    value: branch.into(),
                    bench: bench.clone(),
                    backend_flag: flag.into(),
                }));
            }
            for path in self.paths.iter() {
                runs.push(BenchParam::Path(BenchSettings {
                    identifier: "path".into(),
                    value: path.into(),
                    bench: bench.clone(),
                    backend_flag: flag.into(),
                }));
            }
        }

        runs
    }
}

fn write_bash_file(runs: &[BenchParam], filename: &str, repo: &str) {
    let mut content = String::new();

    for run in runs.iter() {
        content += match run {
            BenchParam::Path(path) => build_bash_path(path),
            BenchParam::Git(git) => build_bash_git(git, repo),
        }
        .as_str();
    }
    content += "cat ";
    for run in runs.iter() {
        content += run.settings().bench_filename().as_str();
        content += " ";
    }

    content += format!("| criterion-table >  {MD_FILENAME}").as_str();

    if let Ok(_) = Command::new("pandoc").arg("--help").output() {
        content += format!("\npandoc -f markdown {MD_FILENAME} > {HTML_FILENAME}").as_str();
    }

    let mut file = File::create(filename).unwrap();
    write!(file, "{}", content).unwrap();
}

fn prepare(runs: &[BenchParam], repo: &str) {
    std::fs::create_dir_all(OUTPUT_DIR).unwrap();

    write_bash_file(runs, SH_FILENAME, repo);
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
        std::fs::remove_file(run.settings().bench_filename()).ok();
    }

    std::fs::remove_file(SH_FILENAME).ok();
    std::fs::remove_file("tables.toml").ok();
}

fn build_bash_git(run: &BenchSettings, repo: &str) -> String {
    let ops = format!("--no-default-features --features {}", run.backend_flag);

    let identifier = &run.identifier;
    let value = &run.value;
    let bench = &run.bench;

    let mut output = String::new();
    let version = run.value.replace('/', "-");
    output += format!("echo {} > {}\n", version, version_file()).as_str();
    output += format!("cargo add burn --git {repo} --{identifier} {value}\n").as_str();
    output += format!("cargo add burn-wgpu --git {repo} --{identifier} {value}\n").as_str();
    output += format!("cargo add burn-tch --git {repo} --{identifier} {value}\n").as_str();
    output += format!("cargo add burn-ndarray --git {repo} --{identifier} {value}\n").as_str();
    output += format!("cargo add burn-autodiff --git {repo} --{identifier} {value}\n").as_str();
    output += format!(
        "cargo criterion {ops} {bench} --message-format=json > {}\n",
        run.bench_filename()
    )
    .as_str();
    println!("Build bash git {}", output);
    output
}

fn build_bash_path(run: &BenchSettings) -> String {
    let ops = format!("--no-default-features --features {}", run.backend_flag);

    let value = &run.value;
    let bench = &run.bench;

    let mut output = String::new();
    let version = run.value.replace('/', "-");
    output += format!("echo {} > {}\n", version, version_file()).as_str();
    output += format!("cargo add burn --path {value}/burn \n").as_str();
    output += format!("cargo add burn-wgpu --path {value}/burn-wgpu\n").as_str();
    output += format!("cargo add burn-tch --path {value}/burn-tch\n").as_str();
    output += format!("cargo add burn-ndarray --path {value}/burn-ndarray\n").as_str();
    output += format!("cargo add burn-autodiff --path {value}/burn-autodiff\n").as_str();
    output += format!(
        "cargo criterion {ops} {bench} --message-format=json > {}\n",
        run.bench_filename()
    )
    .as_str();
    println!("Build bash path {}", output);
    output
}
