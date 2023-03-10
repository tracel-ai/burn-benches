use crate::{bench::BenchSuite, conv2d, mlp, transformer};
use nvml_wrapper::Nvml;
use std::{fs::File, io::Write};
use sysinfo::{CpuExt, System, SystemExt};

pub fn make_tables() {
    let mut file = File::create("tables.toml").unwrap();
    file.write("[top_comments]\n".to_string().as_bytes())
        .unwrap();

    write_section(
        &mut file,
        "Overview",
        format!("Burn micro benchmarks\n{}", system_infos()).as_str(),
    );

    file.write("[table_comments]\n".to_string().as_bytes())
        .unwrap();

    write_bench_suite::<mlp::MlpBenchSuite>(&mut file);
    write_bench_suite::<transformer::TransformerBenchSuite>(&mut file);
    write_bench_suite::<conv2d::Conv2dBenchSuite>(&mut file);
}

fn system_infos() -> String {
    let sys = System::new_all();
    let mut info = String::from("\n## System\n\n");

    sys.name()
        .map(|name| info += format!("- Name: {name}\n").as_str());
    sys.os_version()
        .map(|version| info += format!("- OS version: {version}\n").as_str());
    sys.kernel_version()
        .map(|version| info += format!("- Kernel Version: {version}\n").as_str());
    let cpu = sys.global_cpu_info();
    info += format!("- CPU: {} {}\n", cpu.brand(), cpu.name()).as_str();

    if let Ok(nvml) = Nvml::init() {
        if let Ok(count) = nvml.device_count() {
            for index in 0..count {
                if let Ok(device) = nvml.device_by_index(index) {
                    match count {
                        1 => info += "- GPU: ",
                        _ => info += format!("- GPU({index}): ").as_str(),
                    };
                    info += format!("{:?} {}", device.brand().unwrap(), device.name().unwrap())
                        .as_str();
                }
            }
        }
    }

    info
}

fn write_bench_suite<B: BenchSuite>(file: &mut File) {
    file.write(format!("{} = \"\"\"\n", B::name()).as_bytes())
        .unwrap();
    file.write(B::details().as_bytes()).unwrap();
    file.write("\n\"\"\"\n".to_string().as_bytes()).unwrap();
}

fn write_section(file: &mut File, name: &str, content: &str) {
    file.write(format!("{name} = \"\"\"\n").as_bytes()).unwrap();
    file.write(content.as_bytes()).unwrap();
    file.write("\n\"\"\"\n".to_string().as_bytes()).unwrap();
}
