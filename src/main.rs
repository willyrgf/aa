use std::env;

use aa::{
    experiments::run_experiments,
    task::Task,
    verbosity::{self, parse_verbosity},
};

fn main() {
    let v = parse_verbosity(env::args().skip(1));
    verbosity::init(v);

    let tasks = [Task::Add, Task::Mul, Task::Xor, Task::Nand, Task::KeepX];
    let sample_sizes = [6usize, 10];
    let trials_per_sample = 128usize;

    run_experiments(&tasks, &sample_sizes, trials_per_sample);
}
