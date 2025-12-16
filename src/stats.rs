/// TrialResult contains the pre-computed metrics for a single experiment trial.
/// This decouples the stats module from the internal experiment details.
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// True positive rate (0.0 to 1.0)
    pub tp: f32,
    /// False positive rate (0.0 to 1.0)
    pub fp: f32,
    /// False negative rate (0.0 to 1.0)
    pub fn_: f32,
    /// Weakness score (number of states where policy is weak)
    pub weakness: usize,
    /// Description length of the policy
    pub description_length: usize,
    /// Number of clauses in the policy
    pub num_clauses: usize,
    /// Number of necessary clauses
    pub num_necessary: usize,
}

#[derive(Default, Debug)]
pub struct Stats {
    pub trials: usize,
    pub found: usize,
    pub perfect: usize,
    pub tp_sum: f32,
    pub fp_sum: f32,
    pub fn_sum: f32,
    pub weakness_sum: usize,
    pub length_sum: usize,
    pub clauses_sum: usize,
    pub ext_sum: usize,
    pub nec_sum: usize,
    pub timeouts: usize,
}

impl Stats {
    /// Update stats with the result of a single trial.
    ///
    /// # Arguments
    /// * `result` - The pre-computed trial result, or None if no policy was found
    /// * `timed_out` - Whether the trial timed out
    pub fn update(&mut self, result: Option<TrialResult>, timed_out: bool) {
        self.trials += 1;
        if timed_out {
            self.timeouts += 1;
        }
        if let Some(trial) = result {
            self.found += 1;
            if (trial.tp - 1.0).abs() < f32::EPSILON {
                self.perfect += 1;
            }
            self.tp_sum += trial.tp;
            self.fp_sum += trial.fp;
            self.fn_sum += trial.fn_;
            self.weakness_sum += trial.weakness;
            self.length_sum += trial.description_length;
            self.clauses_sum += trial.num_clauses;
            self.ext_sum += trial.weakness;
            self.nec_sum += trial.num_necessary;
        }
    }

    /// Merge another Stats instance into this one.
    /// Useful for combining results from parallel experiments.
    pub fn merge(&mut self, other: Stats) {
        self.trials += other.trials;
        self.found += other.found;
        self.perfect += other.perfect;
        self.tp_sum += other.tp_sum;
        self.fp_sum += other.fp_sum;
        self.fn_sum += other.fn_sum;
        self.weakness_sum += other.weakness_sum;
        self.length_sum += other.length_sum;
        self.clauses_sum += other.clauses_sum;
        self.ext_sum += other.ext_sum;
        self.nec_sum += other.nec_sum;
        self.timeouts += other.timeouts;
    }
}

/// Print formatted statistics for a policy experiment.
///
/// # Arguments
/// * `label` - A label to display for these statistics
/// * `stats` - The statistics to print
/// * `universe_size` - The size of the universe (default: 256 for 8-bit state space)
pub fn print_policy_stats(label: &str, stats: &Stats, universe_size: f32) {
    println!("{}:", label);
    println!(
        "  found   : {}/{} (perfect {} | Rate {:.3}) timeouts: {}",
        stats.found,
        stats.trials,
        stats.perfect,
        stats.perfect as f32 / stats.trials as f32,
        stats.timeouts
    );

    if stats.found == 0 {
        return;
    }

    let denom = stats.found as f32;
    let avg_ext = stats.ext_sum as f32 / (denom * universe_size);
    println!(
        "  AvgExt={:.3}  avg weakness: {:.2}  avg length: {:.2}  avg clauses: {:.2}  avg nec: {:.2}",
        avg_ext,
        stats.weakness_sum as f32 / denom,
        stats.length_sum as f32 / denom,
        stats.clauses_sum as f32 / denom,
        stats.nec_sum as f32 / denom
    );
    println!(
        "  avg TP={:.3} FP={:.3} FN={:.3}",
        stats.tp_sum / denom,
        stats.fp_sum / denom,
        stats.fn_sum / denom
    );
}
