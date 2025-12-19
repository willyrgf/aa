use crate::{
    experiments::run_experiments,
    persistence::{list_policies, load_policy},
    task::Task,
};
use std::path::PathBuf;

mod cnf;
mod cnf_qm_petrick;
mod dataset;
mod experiments;
mod persistence;
mod rng;
mod search;
mod state;
mod stats;
mod task;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Default: run experiments without saving
    if args.len() == 1 || (args.len() == 2 && args[1] == "run") {
        run_experiment_mode(false);
    } else if args.len() == 3 && args[1] == "run" && args[2] == "--save" {
        run_experiment_mode(true);
    } else if args.len() >= 2 && args[1] == "query" {
        query_mode(&args[2..]);
    } else {
        print_usage();
        std::process::exit(1);
    }
}

fn run_experiment_mode(save: bool) {
    let tasks = [Task::Add, Task::Mul, Task::Xor, Task::Nand, Task::KeepX];
    let sample_sizes = [6usize, 10, 14];
    let trials_per_sample = 256usize;
    let policies_dir = PathBuf::from("policies");

    if save {
        println!("Policy saving: ENABLED");
        println!("Policies directory: {}", policies_dir.display());
        println!();
    } else {
        println!("Policy saving: DISABLED (use 'cargo run -- run --save' to enable)");
        println!();
    }

    run_experiments(
        &tasks,
        &sample_sizes,
        trials_per_sample,
        save,
        &policies_dir,
    );
}

fn query_mode(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: cargo run -- query <policy_file> <state_value>");
        eprintln!("   OR: cargo run -- query --list");
        std::process::exit(1);
    }

    if args[0] == "--list" {
        list_mode();
        return;
    }

    if args.len() < 2 {
        eprintln!("Usage: cargo run -- query <policy_file> <state_value>");
        std::process::exit(1);
    }

    let policy_path = PathBuf::from(&args[0]);
    let state_value: u8 = match args[1].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Error: state_value must be a number 0-255");
            std::process::exit(1);
        }
    };

    match load_policy(&policy_path) {
        Ok(policy) => {
            let state = state::State(state_value);
            let result = policy.eval(state);

            println!("Policy: {}", policy_path.display());
            println!("  Task: {}", policy.task_label);
            println!("  Sample size: {}", policy.sample_size);
            println!("  Objective: {}", policy.objective);
            println!("  Target bit: {}", policy.target_bit);
            println!("  CNF: {}", policy.cnf);
            println!(
                "  Performance: TP={:.3}, FP={:.3}, FN={:.3}",
                policy.tp, policy.fp, policy.fn_
            );
            println!();
            println!("Query: State({})", state_value);
            println!("  {}", state);
            println!("  Result: {}", result);
        }
        Err(e) => {
            eprintln!("Error loading policy: {}", e);
            std::process::exit(1);
        }
    }
}

fn list_mode() {
    let policies_dir = PathBuf::from("policies");

    match list_policies(&policies_dir) {
        Ok(policies) => {
            if policies.is_empty() {
                println!("No policies found in {}", policies_dir.display());
                println!("Run 'cargo run -- run --save' to generate policies.");
            } else {
                println!(
                    "Found {} policies in {}:\n",
                    policies.len(),
                    policies_dir.display()
                );
                for path in policies {
                    match load_policy(&path) {
                        Ok(p) => {
                            println!(
                                "  {} - {} (k={}, obj={}, TP={:.3})",
                                path.file_name().unwrap().to_str().unwrap(),
                                p.task_label,
                                p.sample_size,
                                p.objective,
                                p.tp
                            );
                        }
                        Err(_) => {
                            println!(
                                "  {} - (error loading)",
                                path.file_name().unwrap().to_str().unwrap()
                            );
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error listing policies: {}", e);
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo run                        - Run experiments (no saving)");
    println!("  cargo run -- run                 - Run experiments (no saving)");
    println!("  cargo run -- run --save          - Run experiments and save best policies");
    println!("  cargo run -- query --list        - List all saved policies");
    println!("  cargo run -- query <file> <state> - Query a policy with a state value (0-255)");
    println!();
    println!("Examples:");
    println!("  cargo run -- run --save");
    println!("  cargo run -- query policies/addition_weakness_k6_*.json 133");
    println!("  cargo run -- query --list");
}
