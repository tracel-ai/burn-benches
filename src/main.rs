use burn_benches::cli::{Benches, BenchesArgs};
use clap::Parser;

fn main() {
    let args = BenchesArgs::parse();
    let repo = args.repository.clone();
    let benches = Benches::new(args.into(), repo);

    benches.run();
}
