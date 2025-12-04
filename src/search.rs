use crate::dataset::Dataset;
use crate::four_bits::Bits4;
use crate::policy::{Policy, RulePolicy};

pub fn fits_dataset<P: Policy>(policy: &P, dataset: &Dataset) -> bool {
    dataset
        .samples
        .iter()
        .all(|s| policy.predict(s.input) == s.output)
}

pub fn accuracy_on_inputs<P: Policy, F>(policy: &P, accuracy_fn: F) -> f32
where
    F: Fn(Bits4) -> Bits4,
{
    let runs = 16u8;
    let mut corrected = 0u32;
    for x in 0..runs {
        let y_accu = accuracy_fn(x);
        let y_pred = policy.predict(x);
        if y_accu == y_pred {
            corrected += 1;
        }
    }
    corrected as f32 / runs as f32
}

pub fn naive_candidates_from_dataset(dataset: &Dataset) -> Vec<RulePolicy> {
    let mut policies = Vec::new();

    // one "memorise everything" policy, one rule per sample
    let mut mem = RulePolicy::with_default(0);
    for s in &dataset.samples {
        mem.add_rule(0x0F, s.input, s.output);
    }
    policies.push(mem);

    // one-rule-per-sample policies with some wildcards
    for s in &dataset.samples {
        // for now, just drop the lowest bit as wildcard
        let mut p = RulePolicy::with_default(0);
        let mask = 0b1110; // ignore bit 0
        p.add_rule(mask, s.input, s.output);
        policies.push(p);
    }

    policies
}

pub fn learn_simp_max(dataset: &Dataset, candidates: &[RulePolicy]) -> Option<RulePolicy> {
    candidates
        .iter()
        .filter(|p| fits_dataset(*p, dataset))
        .min_by_key(|p| p.complexity())
        .cloned()
}

pub fn learn_w_max(dataset: &Dataset, candidates: &[RulePolicy]) -> Option<RulePolicy> {
    candidates
        .iter()
        .filter(|p| fits_dataset(*p, dataset))
        .max_by(|a, b| {
            let wa = a.generalisability(dataset);
            let wb = b.generalisability(dataset);
            wa.cmp(&wb)
                .then_with(|| b.complexity().cmp(&a.complexity())) // weaker first, simpler tie-breaker
        })
        .cloned()
}
