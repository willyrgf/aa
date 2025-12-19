use crate::rng::SplitMix64;
use crate::state::{Bits, State, encode};
use crate::task::Task;

// generate_truth_decision_table generate all possible truth states
// for a task based on state size. known as `dn`
pub fn generate_truth_decision_table(task: Task) -> Vec<State> {
    // Each operand uses State::BITS / 4 bits, so can represent 2^(State::BITS/4) values
    let operand_size = 1 << (State::BITS / 4);
    let mut table = Vec::with_capacity(operand_size * operand_size);

    for x in 0..operand_size {
        for y in 0..operand_size {
            let s = encode(x as Bits, y as Bits, Bits::MIN);
            table.push(task.apply(&s));
        }
    }

    table
}

// sample_decision_table_k samples the truth decision table randomly
// by size-k using Fisher-Yates shuffle
pub fn sample_decision_table_k(dn: &[State], k: usize, rng: &mut SplitMix64) -> Vec<State> {
    let n = dn.len();
    let k = k.min(n);
    let mut indexes: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        indexes.swap(i, j);
    }
    indexes[..k].iter().map(|&idx| dn[idx]).collect()
}

#[test]
fn test_generate_truth_decision_table_size() {
    // Each operand uses State::BITS / 4 bits, so can represent 2^(State::BITS/4) values
    let operand_size = 1 << (State::BITS / 4);
    let expected_entries = operand_size * operand_size;
    let tasks = [Task::Add, Task::Mul, Task::Xor, Task::Nand];

    for task in tasks {
        let table = generate_truth_decision_table(task);
        assert_eq!(
            table.len(),
            expected_entries,
            "{} truth table should have {} entries",
            task.label(),
            expected_entries
        );
    }
}

#[test]
fn test_generate_truth_decision_table_add() {
    let table = generate_truth_decision_table(Task::Add);

    // Each operand uses State::BITS / 4 bits
    let operand_size = 1 << (State::BITS / 4);
    // Output uses State::BITS / 2 bits
    let output_mask = (1 << (State::BITS / 2)) - 1;

    let mut index = 0;
    for x in 0..operand_size {
        for y in 0..operand_size {
            let (got_x, got_y, got_z) = table[index].decode();
            let expected_z = ((x + y) & output_mask) as Bits;

            assert_eq!(got_x, x as Bits, "entry {} should have x = {}", index, x);
            assert_eq!(got_y, y as Bits, "entry {} should have y = {}", index, y);
            assert_eq!(
                got_z, expected_z,
                "entry {} (x={}, y={}): expected z={}, got z={}",
                index, x, y, expected_z, got_z
            );

            index += 1;
        }
    }
}

#[test]
fn test_generate_truth_decision_table_mul() {
    let table = generate_truth_decision_table(Task::Mul);

    let operand_size = 1 << (State::BITS / 4);
    let output_mask = (1 << (State::BITS / 2)) - 1;

    let mut index = 0;
    for x in 0..operand_size {
        for y in 0..operand_size {
            let (got_x, got_y, got_z) = table[index].decode();
            let expected_z = ((x * y) & output_mask) as Bits;

            assert_eq!(got_x, x as Bits);
            assert_eq!(got_y, y as Bits);
            assert_eq!(
                got_z, expected_z,
                "mul entry {} (x={}, y={}): expected z={}, got z={}",
                index, x, y, expected_z, got_z
            );

            index += 1;
        }
    }
}

#[test]
fn test_generate_truth_decision_table_xor() {
    let table = generate_truth_decision_table(Task::Xor);

    let operand_size = 1 << (State::BITS / 4);
    let output_mask = (1 << (State::BITS / 2)) - 1;

    let mut index = 0;
    for x in 0..operand_size {
        for y in 0..operand_size {
            let (got_x, got_y, got_z) = table[index].decode();
            let expected_z = ((x ^ y) & output_mask) as Bits;

            assert_eq!(got_x, x as Bits);
            assert_eq!(got_y, y as Bits);
            assert_eq!(
                got_z, expected_z,
                "xor entry {} (x={}, y={}): expected z={}, got z={}",
                index, x, y, expected_z, got_z
            );

            index += 1;
        }
    }
}

#[test]
fn test_generate_truth_decision_table_nand() {
    let table = generate_truth_decision_table(Task::Nand);

    let operand_size = 1 << (State::BITS / 4);
    let output_mask = (1 << (State::BITS / 2)) - 1;

    let mut index = 0;
    for x in 0..operand_size {
        for y in 0..operand_size {
            let (got_x, got_y, got_z) = table[index].decode();
            let expected_z = (!(x & y) & output_mask) as Bits;

            assert_eq!(got_x, x as Bits);
            assert_eq!(got_y, y as Bits);
            assert_eq!(
                got_z, expected_z,
                "nand entry {} (x={}, y={}): expected z={}, got z={}",
                index, x, y, expected_z, got_z
            );

            index += 1;
        }
    }
}

#[test]
fn test_sample_decision_table_k_size() {
    let table = generate_truth_decision_table(Task::Add);
    let mut rng = SplitMix64::new(42);

    // sample k=5 elements from the table
    let sample = sample_decision_table_k(&table, 5, &mut rng);
    assert_eq!(sample.len(), 5, "should sample exactly k elements");

    // sample k=0 elements
    let sample = sample_decision_table_k(&table, 0, &mut rng);
    assert_eq!(sample.len(), 0, "should return empty vec for k=0");

    // sample k larger than table size
    let sample = sample_decision_table_k(&table, table.len() * 2, &mut rng);
    assert_eq!(
        sample.len(),
        table.len(),
        "should sample at most table.len() elements"
    );
}

#[test]
fn test_sample_decision_table_k_subset() {
    let table = generate_truth_decision_table(Task::Add);
    let mut rng = SplitMix64::new(123);

    let sample = sample_decision_table_k(&table, 8, &mut rng);

    // verify all sampled elements exist in the original table
    for state in &sample {
        assert!(
            table.contains(state),
            "sampled element {:?} should exist in original table",
            state
        );
    }
}

#[test]
fn test_sample_decision_table_k_deterministic() {
    let table = generate_truth_decision_table(Task::Xor);

    // same seed should produce same sample
    let mut rng1 = SplitMix64::new(999);
    let sample1 = sample_decision_table_k(&table, 6, &mut rng1);

    let mut rng2 = SplitMix64::new(999);
    let sample2 = sample_decision_table_k(&table, 6, &mut rng2);

    assert_eq!(
        sample1, sample2,
        "same seed should produce identical samples"
    );
}

#[test]
fn test_sample_decision_table_k_different_seeds() {
    let table = generate_truth_decision_table(Task::Mul);

    let mut rng1 = SplitMix64::new(111);
    let sample1 = sample_decision_table_k(&table, 10, &mut rng1);

    let mut rng2 = SplitMix64::new(222);
    let sample2 = sample_decision_table_k(&table, 10, &mut rng2);

    // different seeds should produce different samples (very high probability)
    assert_ne!(
        sample1, sample2,
        "different seeds should produce different samples"
    );
}
