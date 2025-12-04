use crate::dataset::Dataset;
use crate::four_bits::{Bits4, mask4};
use crate::policy::Policy;
use crate::rng::Lcg;
use crate::search::{
    accuracy_on_inputs, learn_simp_max, learn_w_max, naive_candidates_from_dataset,
};

fn ground_truth_add2plus2(x: Bits4) -> Bits4 {
    // interpret lower 2 bits as a, upper 2 bits as b
    let a = x & 0b0011;
    let b = (x & 0b1100) >> 2;
    let sum = a + b;
    mask4(sum)
}

pub fn random_training_set(size: usize, seed: u64) -> Dataset {
    let mut rng = Lcg::new(seed);
    let mut data = Dataset::new();
    for _ in 0..size {
        let x = rng.next_bits4();
        let y = ground_truth_add2plus2(x);
        data.push(x, y);
    }
    data
}

pub fn run_simple_experiment() {
    for n in 1..=8 {
        let dataset = random_training_set(n, 42);
        let candidates = naive_candidates_from_dataset(&dataset);

        let simp = learn_simp_max(&dataset, &candidates);
        let w = learn_w_max(&dataset, &candidates);

        println!("training size = {}", n);

        if let Some(p) = simp {
            let acc = accuracy_on_inputs(&p, ground_truth_add2plus2);
            println!(
                "  simp-max acc = {:.3}, complexity = {}, generalisability = {}",
                acc,
                p.complexity(),
                p.generalisability(&dataset)
            );
        } else {
            println!("  simp-max: no fitting policy");
        }

        if let Some(p) = w {
            let acc = accuracy_on_inputs(&p, ground_truth_add2plus2);
            println!(
                "  w-max   acc = {:.3}, complexity = {}, generalisability = {}",
                acc,
                p.complexity(),
                p.generalisability(&dataset)
            );
        } else {
            println!("  w-max: no fitting policy");
        }
    }
}
