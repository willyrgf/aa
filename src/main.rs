use crate::{experiments::run_experiments, task::Task};

mod cnf;
mod cnf_qm_petrick;
mod dataset;
mod experiments;
mod rng;
mod search;
mod state;
mod stats;
mod task;

fn main() {
    let tasks = [Task::Add, Task::Mul, Task::Xor, Task::Nand];
    let sample_sizes = [6usize, 10];
    let trials_per_sample = 128usize;

    run_experiments(&tasks, &sample_sizes, trials_per_sample);
}
