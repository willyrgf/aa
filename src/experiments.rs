use std::{
    sync::{Arc, Mutex},
    thread,
};

use crate::{
    cnf::{Cnf, accuracy_on_decision, weakness},
    cnf_qm_petrick::simplified_cnf,
    dataset::{generate_truth_decision_table, sample_decision_table_k},
    rng::SplitMix64,
    search::{Objective, best_first_policy, necessary_clauses},
    state::State,
    stats::{Stats, TrialResult, print_policy_stats},
    task::Task,
};

#[derive(Clone, Debug)]
pub struct Trial {
    pub target: u8,
    pub dk: Vec<State>,
}

/// configuration for experiment execution
#[derive(Clone, Debug)]
pub struct ExpConfig {
    pub beam_size: usize,
    pub timeout_ms: u64,
    pub num_threads: usize,
}

impl Default for ExpConfig {
    fn default() -> Self {
        Self {
            beam_size: 16,
            timeout_ms: 60_000, // 60 seconds
            num_threads: thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }
}

// complete context for a single experiment run (one task, one sample size k)
#[derive(Clone, Debug)]
pub struct ExpCtx {
    pub task: Task,
    pub task_label: String,
    pub universe: Vec<State>,
    pub dn: Vec<State>,
    pub sample_size: usize,
    pub trials: Vec<Trial>,
    pub config: ExpConfig,
}

// generate all experiment contexts for the given tasks and sample sizes
pub fn generate_experiment_contexts(
    tasks: &[Task],
    sample_sizes: &[usize],
    trials_per_sample: usize,
    config: ExpConfig,
) -> Vec<ExpCtx> {
    let base_seed = 123_456_789u64;
    let universe: Vec<State> = State::universe().collect();
    let mut contexts = Vec::new();

    for (task_idx, task) in tasks.iter().enumerate() {
        let seed = base_seed + task_idx as u64;
        let mut rng = SplitMix64::new(seed);
        let dn = generate_truth_decision_table(*task);

        for &k in sample_sizes {
            let mut trials = Vec::new();
            for _ in 0..trials_per_sample {
                let target = rng.next_usize(State::BITS as usize) as u8;
                let dk = sample_decision_table_k(&dn, k, &mut rng);
                trials.push(Trial { target, dk });
            }

            contexts.push(ExpCtx {
                task: *task,
                task_label: task.label().to_string(),
                universe: universe.clone(),
                dn: dn.clone(),
                sample_size: k,
                trials,
                config: config.clone(),
            });
        }
    }

    contexts
}

fn policy_to_trial_result(
    policy: Option<Cnf>,
    universe: &[State],
    dn: &[State],
    target: u8,
) -> Option<TrialResult> {
    policy.map(|cnf| {
        let (tp, fp, fn_) = accuracy_on_decision(&cnf, universe, dn, target);
        let weak = weakness(&cnf, universe);
        let num_necessary = necessary_clauses(&cnf, universe, dn, target).len();

        TrialResult {
            tp,
            fp,
            fn_,
            weakness: weak,
            description_length: cnf.description_length(),
            num_clauses: cnf.len(),
            num_necessary,
        }
    })
}

// result of executing an experiment context
#[derive(Debug)]
pub struct ExpResult {
    pub task_label: String,
    pub sample_size: usize,
    pub weakness_stats: Stats,
    pub simplicity_stats: Stats,
}

// execute a single experiment context in parallel
pub fn execute_experiment(ctx: &ExpCtx) -> ExpResult {
    let num_threads = ctx.config.num_threads;

    let universe = Arc::new(ctx.universe.clone());
    let dn = Arc::new(ctx.dn.clone());
    let w_stats = Arc::new(Mutex::new(Stats::default()));
    let s_stats = Arc::new(Mutex::new(Stats::default()));

    let chunk_size = (ctx.trials.len() + num_threads - 1) / num_threads;
    let mut handles = vec![];

    for chunk in ctx.trials.chunks(chunk_size) {
        let chunk = chunk.to_vec();
        let universe = Arc::clone(&universe);
        let dn = Arc::clone(&dn);
        let w_stats = Arc::clone(&w_stats);
        let s_stats = Arc::clone(&s_stats);
        let beam_size = ctx.config.beam_size;
        let timeout_ms = ctx.config.timeout_ms;

        let handle = thread::spawn(move || {
            let mut local_w_stats = Stats::default();
            let mut local_s_stats = Stats::default();

            for trial in chunk {
                let base_cnf = simplified_cnf(&trial.dk, &universe, trial.target);

                let (w_policy, w_timeout) = best_first_policy(
                    &base_cnf,
                    &universe,
                    &trial.dk,
                    trial.target,
                    Objective::Weakness,
                    beam_size,
                    timeout_ms as u128,
                );
                let w_result = policy_to_trial_result(w_policy, &universe, &dn, trial.target);
                local_w_stats.update(w_result, w_timeout);

                let (s_policy, s_timeout) = best_first_policy(
                    &base_cnf,
                    &universe,
                    &trial.dk,
                    trial.target,
                    Objective::Simplicity,
                    beam_size,
                    timeout_ms as u128,
                );
                let s_result = policy_to_trial_result(s_policy, &universe, &dn, trial.target);
                local_s_stats.update(s_result, s_timeout);
            }

            // merge local stats into global stats
            w_stats.lock().unwrap().merge(local_w_stats);
            s_stats.lock().unwrap().merge(local_s_stats);
        });

        handles.push(handle);
    }

    // wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let weakness_stats = Arc::try_unwrap(w_stats).unwrap().into_inner().unwrap();
    let simplicity_stats = Arc::try_unwrap(s_stats).unwrap().into_inner().unwrap();

    ExpResult {
        task_label: ctx.task_label.clone(),
        sample_size: ctx.sample_size,
        weakness_stats,
        simplicity_stats,
    }
}

pub fn run_experiments(tasks: &[Task], sample_sizes: &[usize], trials_per_sample: usize) {
    let universe_size = State::universe().count();
    let config = ExpConfig::default();

    println!("Universe size: {}", universe_size);
    println!("Trials per k : {}", trials_per_sample);
    println!("Threads      : {}", config.num_threads);

    // Phase 1: Generate all experiment contexts
    let contexts = generate_experiment_contexts(tasks, sample_sizes, trials_per_sample, config);

    println!("\nGenerated {} experiment contexts", contexts.len());

    // Phase 2: Execute each experiment context
    let mut current_task = String::new();
    for ctx in &contexts {
        // Print task header when we encounter a new task
        if ctx.task_label != current_task {
            current_task = ctx.task_label.clone();
            println!("\n-- Task: {} --", ctx.task_label);
            println!("D_n size     : {}", ctx.dn.len());
        }

        // Execute the experiment
        let result = execute_experiment(ctx);

        // Print results
        println!("\n=== |D_k| = {} ===", result.sample_size);
        print_policy_stats("w-max", &result.weakness_stats, universe_size as f32);
        print_policy_stats("simp-max", &result.simplicity_stats, universe_size as f32);
    }
}
