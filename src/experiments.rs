use std::{
    sync::{Arc, Mutex},
    thread,
};

use crate::{
    cnf::{accuracy_on_decision, weakness, Cnf},
    cnf_qm_petrick::simplified_cnf,
    dataset::{generate_truth_decision_table, sample_decision_table_k},
    rng::SplitMix64,
    search::{best_first_policy, necessary_clauses, Objective},
    state::State,
    stats::{print_policy_stats, Stats, TrialResult},
    task::Task,
};

struct ExpCtx<'a> {
    dn: &'a [State],
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

pub fn run_experiments(tasks: &[Task], sample_sizes: &[usize], trials_per_sample: usize) {
    let base_seed = 123_456_789u64;
    let universe: Vec<State> = State::universe().collect();
    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("Universe size: {}", universe.len());
    println!("Trials per k : {}", trials_per_sample);
    println!("Threads      : {}", num_threads);

    for (task_idx, task) in tasks.iter().enumerate() {
        let seed = base_seed + task_idx as u64;
        let mut rng = SplitMix64::new(seed);

        let dn = generate_truth_decision_table(*task);
        println!("\n-- Task: {} --", task.label());
        println!("D_n size     : {}", dn.len());

        for &k in sample_sizes {
            let mut trial_parameters = Vec::new();
            for _ in 0..trials_per_sample {
                let target = rng.next_usize(State::BITS as usize) as u8;
                let dk = sample_decision_table_k(&dn, k, &mut rng);
                trial_parameters.push((target, dk));
            }

            let universe = Arc::new(universe.clone());
            let dn = Arc::new(dn.clone());
            let w_stats = Arc::new(Mutex::new(Stats::default()));
            let s_stats = Arc::new(Mutex::new(Stats::default()));

            let chunk_size = (trials_per_sample + num_threads - 1) / num_threads;
            let mut handles = vec![];

            for chunk in trial_parameters.chunks(chunk_size) {
                let chunk = chunk.to_vec();
                let universe = Arc::clone(&universe);
                let dn = Arc::clone(&dn);
                let w_stats = Arc::clone(&w_stats);
                let s_stats = Arc::clone(&s_stats);

                let handle = thread::spawn(move || {
                    let mut local_w_stats = Stats::default();
                    let mut local_s_stats = Stats::default();

                    for (target, dk) in chunk {
                        let base_cnf = simplified_cnf(&dk, &universe, target);

                        let (w_policy, w_timeout) = best_first_policy(
                            &base_cnf,
                            &universe,
                            &dk,
                            target,
                            Objective::Weakness,
                            16,
                            60_000, // 60 seconds
                        );
                        let (s_policy, s_timeout) = best_first_policy(
                            &base_cnf,
                            &universe,
                            &dk,
                            target,
                            Objective::Simplicity,
                            16,
                            60_000, // 60 seconds
                        );

                        let w_result = policy_to_trial_result(w_policy, &universe, &dn, target);
                        local_w_stats.update(w_result, w_timeout);

                        let s_result = policy_to_trial_result(s_policy, &universe, &dn, target);
                        local_s_stats.update(s_result, s_timeout);
                    }

                    // Merge local stats into global stats
                    w_stats.lock().unwrap().merge(local_w_stats);
                    s_stats.lock().unwrap().merge(local_s_stats);
                });

                handles.push(handle);
            }

            // Wait for all threads to complete
            for handle in handles {
                handle.join().unwrap();
            }

            let w_stats = Arc::try_unwrap(w_stats).unwrap().into_inner().unwrap();
            let s_stats = Arc::try_unwrap(s_stats).unwrap().into_inner().unwrap();

            println!("\n=== |D_k| = {} ===", k);
            print_policy_stats("w-max", &w_stats, universe.len() as f32);
            print_policy_stats("simp-max", &s_stats, universe.len() as f32);
        }
    }
}
