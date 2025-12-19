use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cnf::Cnf;
use crate::search::Objective;
use crate::state::State;
use crate::stats::TrialResult;

/// SavedPolicy contains a CNF policy and all its metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedPolicy {
    // Core identity
    pub task_label: String,
    pub sample_size: usize,
    pub objective: String,

    // CNF policy
    pub cnf: Cnf,

    // Performance metrics
    pub tp: f32,
    pub fp: f32,
    pub fn_: f32,
    pub weakness: usize,
    pub description_length: usize,
    pub num_clauses: usize,
    pub num_necessary: usize,

    // Context
    pub target_bit: u8,
    pub universe_size: usize,
    pub decision_table_size: usize,

    // Provenance
    pub timestamp: u64,
    pub version: String,
}

impl SavedPolicy {
    /// Create a new SavedPolicy from experiment components
    pub fn new(
        task_label: String,
        sample_size: usize,
        objective: Objective,
        cnf: Cnf,
        trial_result: &TrialResult,
        target_bit: u8,
        universe_size: usize,
        decision_table_size: usize,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            task_label,
            sample_size,
            objective: format!("{:?}", objective).to_lowercase(),
            cnf,
            tp: trial_result.tp,
            fp: trial_result.fp,
            fn_: trial_result.fn_,
            weakness: trial_result.weakness,
            description_length: trial_result.description_length,
            num_clauses: trial_result.num_clauses,
            num_necessary: trial_result.num_necessary,
            target_bit,
            universe_size,
            decision_table_size,
            timestamp,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Evaluate the policy on a given state
    pub fn eval(&self, state: State) -> bool {
        self.cnf.eval(state)
    }
}

/// Save policy to JSON file
pub fn save_policy(policy: &SavedPolicy, policies_dir: &Path) -> Result<PathBuf, std::io::Error> {
    fs::create_dir_all(policies_dir)?;

    let filename = format!(
        "{}_{}_k{}_{}.json",
        policy.task_label.replace(' ', "_"),
        policy.objective,
        policy.sample_size,
        policy.timestamp
    );

    let path = policies_dir.join(filename);
    let file = File::create(&path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, policy)?;

    Ok(path)
}

/// Load policy from JSON file
pub fn load_policy(path: &Path) -> Result<SavedPolicy, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let policy = serde_json::from_reader(reader)?;
    Ok(policy)
}

/// List all policy files in directory
pub fn list_policies(policies_dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut policies = Vec::new();

    if !policies_dir.exists() {
        return Ok(policies);
    }

    for entry in fs::read_dir(policies_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            policies.push(path);
        }
    }

    policies.sort();
    Ok(policies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnf::{Clause, Literal};
    use std::env;

    #[test]
    fn test_save_and_load_policy() {
        let temp_dir = env::temp_dir().join("aa_test_policies");
        fs::create_dir_all(&temp_dir).unwrap();

        // Create test policy
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let trial_result = TrialResult {
            tp: 1.0,
            fp: 0.0,
            fn_: 0.0,
            weakness: 128,
            description_length: 4,
            num_clauses: 1,
            num_necessary: 1,
        };

        let policy = SavedPolicy::new(
            "test".to_string(),
            6,
            Objective::Weakness,
            cnf.clone(),
            &trial_result,
            0,
            256,
            50,
        );

        // Save
        let path = save_policy(&policy, &temp_dir).unwrap();
        assert!(path.exists());

        // Load
        let loaded = load_policy(&path).unwrap();
        assert_eq!(loaded.task_label, "test");
        assert_eq!(loaded.sample_size, 6);
        assert_eq!(loaded.objective, "weakness");
        assert_eq!(loaded.tp, 1.0);

        // Cleanup
        fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_policy_eval() {
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let trial_result = TrialResult {
            tp: 1.0,
            fp: 0.0,
            fn_: 0.0,
            weakness: 128,
            description_length: 4,
            num_clauses: 1,
            num_necessary: 1,
        };

        let policy = SavedPolicy::new(
            "test".to_string(),
            6,
            Objective::Weakness,
            cnf,
            &trial_result,
            0,
            256,
            50,
        );

        // Test evaluation
        assert!(policy.eval(State(0b00000001))); // bit 0 set
        assert!(!policy.eval(State(0b00000000))); // bit 0 not set
    }

    #[test]
    fn test_list_policies() {
        let temp_dir = env::temp_dir().join("aa_test_list_policies");
        fs::create_dir_all(&temp_dir).unwrap();

        // Create test policies
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let trial_result = TrialResult {
            tp: 1.0,
            fp: 0.0,
            fn_: 0.0,
            weakness: 128,
            description_length: 4,
            num_clauses: 1,
            num_necessary: 1,
        };

        let policy1 = SavedPolicy::new(
            "test1".to_string(),
            6,
            Objective::Weakness,
            cnf.clone(),
            &trial_result,
            0,
            256,
            50,
        );

        let policy2 = SavedPolicy::new(
            "test2".to_string(),
            10,
            Objective::Simplicity,
            cnf,
            &trial_result,
            1,
            256,
            100,
        );

        save_policy(&policy1, &temp_dir).unwrap();
        save_policy(&policy2, &temp_dir).unwrap();

        // List
        let policies = list_policies(&temp_dir).unwrap();
        assert_eq!(policies.len(), 2);

        // Cleanup
        fs::remove_dir_all(&temp_dir).unwrap();
    }
}
