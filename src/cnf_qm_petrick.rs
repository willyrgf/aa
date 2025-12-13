use std::collections::HashSet;

use crate::{
    cnf::{Clause, Cnf, Literal},
    state::{Bits, State},
};

// QM (Quine–McCluskey)
// combine two implicants that differ in exactly one care bit into a more general implicant where that bit becomes dont-care

// Term is a specific state pattern and its mask
// bits = 0b1010, mask = 0b0000 means:
//  bit3=1; bit2=0; bit1=1; bit0=0 (no mask)
// bits = 0b1010, mask = 0b0010 means:
//  bit3=1; bit2=0; bit1=dont-care; bit0=0 (bit1 masked out)
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Term {
    bits: Bits, // underlying bit patterns
    mask: Bits, // 1 bits are dont-cares
}

impl Term {
    pub fn care_bits(&self) -> Bits {
        self.bits & !self.mask
    }
    pub fn count_care_bits(&self) -> usize {
        self.care_bits().count_ones() as usize
    }
}

// combine_terms
// * only same mask terms are combined
// * only combine terms that differ in only 1 bit
// example:
//  a.bits = 1010; a.mask = 0000
//  b.bits = 1000; b.mask = 0000
//  diff = 1010 & 1000 = 0010
//  diff & a.mask = 0
//  a.bits & !diff = 1000
//  a.mask | diff = 0010
pub fn combine_terms(a: &Term, b: &Term) -> Option<Term> {
    // if masks are different, they would differ in more than one bit
    if a.mask != b.mask {
        return None;
    }

    // xor will set 1 for any different bits
    let diff = a.bits ^ b.bits;

    // only one bit diff
    if diff.count_ones() != 1 {
        return None;
    }

    // the bit they differ must be a care bit, not a dont-care
    if diff & a.mask != 0 {
        return None;
    }

    let bits_cared = a.bits & !diff;
    let mask_added_dont_care = a.mask | diff;

    Some(Term {
        bits: bits_cared,
        mask: mask_added_dont_care,
    })
}

// term_covers is true if a term and state are equal in all
// bits that the terms care about
pub fn term_covers(term: &Term, state: State) -> bool {
    let term_state = Term {
        bits: state.0,
        mask: term.mask,
    };
    term_state.care_bits() == term.care_bits()
}

// qm_prime_implicants finds a compact representation of the false states
// in terms of prime implicants (which cant be more generalised).
// first step into turning a brute list of false states into a
// compact symbolic description (like a CNF).
// its like finding the "most false" parts of the false states.
pub fn qm_prime_implicants(false_states: &[State]) -> Vec<Term> {
    // groups with count of ones from 0 to 8
    let possible_groups = (State::BITS + 1) as usize;

    // grouping by numbers of ones in false state
    // start with false states as terms with no dont-cares (mask = 0)
    let mut base_groups: Vec<Vec<Term>> = vec![Vec::new(); possible_groups];
    for state in false_states {
        let ones = state.ones();
        let fs_term = Term {
            bits: state.bits(),
            mask: 0,
        };
        base_groups[ones].push(fs_term);
    }

    let mut prime_implicants: Vec<Term> = Vec::new();
    let mut current = base_groups;

    // compare terms between group k and k+1
    // and merging adjacent minterms into larger implicants
    loop {
        let mut next: Vec<Vec<Term>> = vec![Vec::new(); possible_groups];
        let mut combined_set: HashSet<Term> = HashSet::new();

        for idx in 0..current.len().saturating_sub(1) {
            for a in &current[idx] {
                for b in &current[idx + 1] {
                    if let Some(combined) = combine_terms(a, b) {
                        combined_set.insert(a.clone());
                        combined_set.insert(b.clone());
                        let ones = combined.count_care_bits();
                        if !next[ones].contains(&combined) {
                            next[ones].push(combined);
                        }
                    }
                }
            }
        }

        // anything that couldnt be combined is a prime implicant
        // any term that isnt in combined could not be generalized
        // with any neighbor, so it’s prime at this level
        for bucket in &current {
            for term in bucket {
                if !combined_set.contains(&term) && !prime_implicants.contains(&term) {
                    prime_implicants.push(term.clone());
                }
            }
        }

        if combined_set.is_empty() {
            break;
        }

        current = next;
    }

    prime_implicants
}

fn cost_of_combo(combo: &[usize], primes: &[Term]) -> (usize, usize) {
    // total number of literals across all those clauses
    let total_literals: usize = combo.iter().map(|&idx| primes[idx].count_care_bits()).sum();

    // count of implicants clauses in the solution
    let implicants = combo.len();

    (implicants, total_literals)
}

pub fn term_to_clause(term: Term) -> Clause {
    let mut literals: Vec<Literal> = Vec::new();
    for var in 0u8..(State::BITS as u8) {
        if (term.mask >> var) & 1 == 1 {
            // a dont-care position
            continue;
        }
        let bit_is_one = ((term.bits >> var) & 1) != 0;
        literals.push(Literal {
            var,
            neg: bit_is_one,
        });
    }
    Clause(literals)
}

// petrick finds a globally optimal combination of remaining
// prime implicants that covers the remaining minterms,
// according to your cost function
fn petrick(coverage: &[Vec<usize>], primes: &[Term]) -> Vec<usize> {
    let mut products: Vec<Vec<usize>> = vec![Vec::new()];

    // iterate over list of prime indices that can cover one
    // particular minterm
    for options in coverage {
        let mut new_products: Vec<Vec<usize>> = Vec::new();
        for product in &products {
            for &opt in options {
                let mut combo = product.clone();
                if !combo.contains(&opt) {
                    combo.push(opt);
                    combo.sort_unstable();
                }
                new_products.push(combo);
            }
        }

        new_products.sort();
        new_products.dedup();

        let mut best: Option<(usize, usize)> = None;
        let mut pruned: Vec<Vec<usize>> = Vec::new();
        for combo in new_products {
            let cost = cost_of_combo(&combo, primes);
            match best {
                None => {
                    best = Some(cost);
                    pruned.push(combo);
                }
                Some(b) => {
                    if cost < b {
                        best = Some(cost);
                        pruned.clear();
                        pruned.push(combo);
                    } else if cost == b {
                        pruned.push(combo);
                    }
                }
            }
        }

        products = pruned;
    }

    products
        .into_iter()
        .min_by_key(|combo| cost_of_combo(combo, primes))
        .unwrap_or_default()
}

fn select_implicants(false_states: &[State], primes: &[Term]) -> Vec<Term> {
    let mut remaining: Vec<State> = false_states.to_vec();

    // build coverage matrix
    let mut coverage: Vec<Vec<usize>> = Vec::new();
    for &s in &remaining {
        let options: Vec<usize> = primes
            .iter()
            .enumerate()
            .filter_map(|(idx, prime)| {
                if term_covers(prime, s) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        coverage.push(options);
    }

    let mut selected: Vec<usize> = Vec::new();

    loop {
        let mut added = false;
        for opts in &coverage {
            if opts.len() == 1 {
                let only = opts[0];
                if !selected.contains(&only) {
                    selected.push(only);
                }
                added = true;
            }
        }

        if !added {
            break;
        }

        let mut new_remaining = Vec::new();
        let mut new_coverage = Vec::new();
        for (idx, &s) in remaining.iter().enumerate() {
            let covered = selected.iter().any(|&si| term_covers(&primes[si], s));
            if covered {
                continue;
            }
            new_remaining.push(s);
            new_coverage.push(coverage[idx].clone());
        }
        remaining = new_remaining;
        coverage = new_coverage;
    }

    let recover_selected_primes = |selected_idxs: &mut Vec<usize>| -> Vec<Term> {
        selected_idxs.sort_unstable();
        selected_idxs.dedup();
        selected_idxs.iter().map(|&idx| primes[idx]).collect()
    };

    // if every false state got covered by essential primes
    if remaining.is_empty() {
        return recover_selected_primes(&mut selected);
    }

    // if some minterms are still uncovered after stripping essentials
    let extra = petrick(&coverage, primes);
    for idx in extra {
        if !selected.contains(&idx) {
            selected.push(idx);
        }
    }
    recover_selected_primes(&mut selected)
}

pub fn simplified_cnf(positives: &[State], universe: &[State], target: u8) -> Cnf {
    let positives_set: HashSet<State> = positives.iter().copied().collect();
    let false_states: Vec<State> = universe
        .iter()
        .copied()
        .filter(|s| !positives_set.contains(s))
        .collect();

    if false_states.is_empty() {
        return Cnf(Vec::new());
    }

    let primes = qm_prime_implicants(&false_states);
    let implicants = select_implicants(&false_states, &primes);
    let mut clauses: Vec<Clause> = implicants.into_iter().map(term_to_clause).collect();

    clauses.retain(|clause| clause.literals().iter().any(|lit| lit.var == target));

    Cnf(clauses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_care_bits() {
        // no mask, all bits are care bits
        let term = Term {
            bits: 0b1010,
            mask: 0b0000,
        };
        assert_eq!(term.care_bits(), 0b1010);

        // mask out bit 1
        let term = Term {
            bits: 0b1010,
            mask: 0b0010,
        };
        assert_eq!(term.care_bits(), 0b1000);

        // mask out multiple bits
        let term = Term {
            bits: 0b1111,
            mask: 0b0101,
        };
        assert_eq!(term.care_bits(), 0b1010);
    }

    #[test]
    fn test_count_care_bits() {
        let term = Term {
            bits: 0b1010,
            mask: 0b0000,
        };
        assert_eq!(term.count_care_bits(), 2);

        let term = Term {
            bits: 0b1111,
            mask: 0b0101,
        };
        assert_eq!(term.count_care_bits(), 2);

        let term = Term {
            bits: 0b0000,
            mask: 0b1111,
        };
        assert_eq!(term.count_care_bits(), 0);
    }

    #[test]
    fn test_combine_terms_success() {
        // 1010 and 1000 differ only in bit 1
        let a = Term {
            bits: 0b1010,
            mask: 0b0000,
        };
        let b = Term {
            bits: 0b1000,
            mask: 0b0000,
        };

        let result = combine_terms(&a, &b);
        assert!(result.is_some());

        let combined = result.unwrap();
        assert_eq!(combined.bits, 0b1000);
        assert_eq!(combined.mask, 0b0010);
        assert_eq!(combined.care_bits(), 0b1000);
    }

    #[test]
    fn test_combine_terms_different_masks() {
        let a = Term {
            bits: 0b1010,
            mask: 0b0001,
        };
        let b = Term {
            bits: 0b1000,
            mask: 0b0010,
        };

        let result = combine_terms(&a, &b);
        assert!(result.is_none());
    }

    #[test]
    fn test_combine_terms_multiple_bit_diff() {
        // 1010 and 0101 differ in multiple bits
        let a = Term {
            bits: 0b1010,
            mask: 0b0000,
        };
        let b = Term {
            bits: 0b0101,
            mask: 0b0000,
        };

        let result = combine_terms(&a, &b);
        assert!(result.is_none());
    }

    #[test]
    fn test_combine_terms_identical() {
        let a = Term {
            bits: 0b1010,
            mask: 0b0000,
        };
        let b = Term {
            bits: 0b1010,
            mask: 0b0000,
        };

        let result = combine_terms(&a, &b);
        assert!(result.is_none());
    }

    #[test]
    fn test_combine_terms_diff_in_dont_care() {
        // both have same mask, but differ in a dont-care bit
        let a = Term {
            bits: 0b1010,
            mask: 0b0010,
        };
        let b = Term {
            bits: 0b1000,
            mask: 0b0010,
        };

        // they differ in bit 1, which is already a dont-care
        let result = combine_terms(&a, &b);
        assert!(result.is_none());
    }

    #[test]
    fn test_term_covers_matching() {
        let term = Term {
            bits: 0b1010,
            mask: 0b0000,
        };
        let state = State(0b1010);

        assert!(term_covers(&term, state));
    }

    #[test]
    fn test_term_covers_with_dont_care() {
        // term 10-0 (bit 1 is dont-care)
        let term = Term {
            bits: 0b1000,
            mask: 0b0010,
        };

        // should cover both 1010 and 1000
        assert!(term_covers(&term, State(0b1010)));
        assert!(term_covers(&term, State(0b1000)));

        // should not cover 0010 or 0000
        assert!(!term_covers(&term, State(0b0010)));
        assert!(!term_covers(&term, State(0b0000)));
    }

    #[test]
    fn test_term_covers_not_matching() {
        let term = Term {
            bits: 0b1010,
            mask: 0b0000,
        };
        let state = State(0b1000);

        assert!(!term_covers(&term, state));
    }

    #[test]
    fn test_qm_prime_implicants_simple() {
        // false states: 0b00, 0b01
        // these should combine to 0b0- (bit 0 is dont-care)
        let false_states = vec![State(0b00), State(0b01)];

        let primes = qm_prime_implicants(&false_states);

        // should have 1 prime implicant: 0b00 with mask 0b01
        assert_eq!(primes.len(), 1);
        assert_eq!(primes[0].bits, 0b00);
        assert_eq!(primes[0].mask, 0b01);
    }

    #[test]
    fn test_qm_prime_implicants_no_combination() {
        // false states that differ in more than one bit
        let false_states = vec![State(0b0000), State(0b1111)];

        let primes = qm_prime_implicants(&false_states);

        // should have 2 prime implicants (no combination possible)
        assert_eq!(primes.len(), 2);
    }

    #[test]
    fn test_qm_prime_implicants_multiple_levels() {
        // false states: 0b000, 0b001, 0b010, 0b011
        // 000 + 001 -> 00-
        // 010 + 011 -> 01-
        // 00- + 01- -> 0--
        let false_states = vec![State(0b000), State(0b001), State(0b010), State(0b011)];

        let primes = qm_prime_implicants(&false_states);

        // should result in one prime implicant: 0b0-- (only bit 2 matters, must be 0)
        assert_eq!(primes.len(), 1);
        assert_eq!(primes[0].care_bits(), 0b000);
        assert_eq!(primes[0].mask, 0b011);
    }

    #[test]
    fn test_qm_prime_implicants_complex() {
        // a more complex example with partial coverage
        // false states: 0b100, 0b101, 0b110
        // 100 + 101 -> 10-
        // 110 cannot combine with the result
        let false_states = vec![State(0b100), State(0b101), State(0b110)];

        let primes = qm_prime_implicants(&false_states);

        // should have 2 prime implicants: 10- and 110
        assert_eq!(primes.len(), 2);
    }

    #[test]
    fn test_cost_of_combo_single_term() {
        let primes = vec![Term {
            bits: 0b1010,
            mask: 0b0000,
        }];
        let combo = vec![0];

        let (implicants, literals) = cost_of_combo(&combo, &primes);

        assert_eq!(implicants, 1);
        assert_eq!(literals, 2); // two care bits (bits 1 and 3)
    }

    #[test]
    fn test_cost_of_combo_multiple_terms() {
        let primes = vec![
            Term {
                bits: 0b1010,
                mask: 0b0000,
            }, // 2 care bits
            Term {
                bits: 0b1100,
                mask: 0b0001,
            }, // 2 care bits
            Term {
                bits: 0b0000,
                mask: 0b1111,
            }, // 0 care bits
        ];
        let combo = vec![0, 1, 2];

        let (implicants, literals) = cost_of_combo(&combo, &primes);

        assert_eq!(implicants, 3);
        assert_eq!(literals, 4); // 2 + 2 + 0
    }

    #[test]
    fn test_cost_of_combo_empty() {
        let primes = vec![Term {
            bits: 0b1010,
            mask: 0b0000,
        }];
        let combo = vec![];

        let (implicants, literals) = cost_of_combo(&combo, &primes);

        assert_eq!(implicants, 0);
        assert_eq!(literals, 0);
    }

    #[test]
    fn test_term_to_clause_no_mask() {
        // term 1010 with no mask -> all bits are care bits
        let term = Term {
            bits: 0b1010,
            mask: 0b0000,
        };

        let clause = term_to_clause(term);

        // should have 8 literals (for 8-bit state)
        assert_eq!(clause.literals().len(), 8);

        // bit 0 is 0, so literal should be (var=0, neg=false)
        // bit 1 is 1, so literal should be (var=1, neg=true)
        // bit 2 is 0, so literal should be (var=2, neg=false)
        // bit 3 is 1, so literal should be (var=3, neg=true)
        // bits 4-7 are 0, so literals should be (var=4-7, neg=false)
        let lits = clause.literals();
        assert_eq!(lits[0].var, 0);
        assert_eq!(lits[0].neg, false);
        assert_eq!(lits[1].var, 1);
        assert_eq!(lits[1].neg, true);
        assert_eq!(lits[2].var, 2);
        assert_eq!(lits[2].neg, false);
        assert_eq!(lits[3].var, 3);
        assert_eq!(lits[3].neg, true);
        for i in 4..8 {
            assert_eq!(lits[i].var, i as u8);
            assert_eq!(lits[i].neg, false);
        }
    }

    #[test]
    fn test_term_to_clause_with_mask() {
        // term 1010 with mask 0010 -> bit 1 is dont-care
        let term = Term {
            bits: 0b1010,
            mask: 0b0010,
        };

        let clause = term_to_clause(term);

        // should have 7 literals (8 bits - 1 dont-care)
        assert_eq!(clause.literals().len(), 7);

        let lits = clause.literals();
        // bit 0 is 0, so literal should be (var=0, neg=false)
        assert_eq!(lits[0].var, 0);
        assert_eq!(lits[0].neg, false);
        // bit 1 is dont-care, so it should be skipped
        // bit 2 is 0, so literal should be (var=2, neg=false)
        assert_eq!(lits[1].var, 2);
        assert_eq!(lits[1].neg, false);
        // bit 3 is 1, so literal should be (var=3, neg=true)
        assert_eq!(lits[2].var, 3);
        assert_eq!(lits[2].neg, true);
        // bits 4-7 are 0, so literals should be (var=4-7, neg=false)
        for i in 3..7 {
            assert_eq!(lits[i].var, (i + 1) as u8);
            assert_eq!(lits[i].neg, false);
        }
    }

    #[test]
    fn test_term_to_clause_all_dont_care() {
        // all bits are dont-care (all 8 bits)
        let term = Term {
            bits: 0b0000,
            mask: 0b11111111,
        };

        let clause = term_to_clause(term);

        // should have 0 literals
        assert_eq!(clause.literals().len(), 0);
    }

    #[test]
    fn test_select_implicants_all_essential() {
        // create a scenario where each false state is covered by exactly one prime
        let false_states = vec![State(0b00), State(0b11)];
        let primes = vec![
            Term {
                bits: 0b00,
                mask: 0b00,
            },
            Term {
                bits: 0b11,
                mask: 0b00,
            },
        ];

        let selected = select_implicants(&false_states, &primes);

        // both primes should be selected as essential
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&primes[0]));
        assert!(selected.contains(&primes[1]));
    }

    #[test]
    fn test_select_implicants_with_overlap() {
        // false states that can be covered by overlapping primes
        let false_states = vec![State(0b00), State(0b01)];
        let primes = vec![
            Term {
                bits: 0b00,
                mask: 0b00,
            }, // covers 00
            Term {
                bits: 0b01,
                mask: 0b00,
            }, // covers 01
            Term {
                bits: 0b00,
                mask: 0b01,
            }, // covers both 00 and 01
        ];

        let selected = select_implicants(&false_states, &primes);

        // should select the most general prime that covers both
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].mask, 0b01);
    }

    #[test]
    fn test_petrick_simple() {
        // simple coverage where each minterm has one option
        let primes = vec![
            Term {
                bits: 0b00,
                mask: 0b00,
            },
            Term {
                bits: 0b11,
                mask: 0b00,
            },
        ];
        let coverage = vec![vec![0], vec![1]];

        let result = petrick(&coverage, &primes);

        // should select both primes
        assert_eq!(result.len(), 2);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }

    #[test]
    fn test_petrick_with_choice() {
        // minterm can be covered by multiple primes, petrick should pick the better one
        let primes = vec![
            Term {
                bits: 0b0000,
                mask: 0b0000,
            }, // 0 care bits
            Term {
                bits: 0b1111,
                mask: 0b0000,
            }, // 4 care bits
            Term {
                bits: 0b0000,
                mask: 0b1111,
            }, // 0 care bits (all dont-care)
        ];
        let coverage = vec![
            vec![0, 2], // first minterm can be covered by prime 0 or 2
        ];

        let result = petrick(&coverage, &primes);

        // should select the prime with lower cost
        assert_eq!(result.len(), 1);
        assert!(result.contains(&0) || result.contains(&2));
    }

    #[test]
    fn test_petrick_empty_coverage() {
        let primes = vec![Term {
            bits: 0b00,
            mask: 0b00,
        }];
        let coverage: Vec<Vec<usize>> = vec![];

        let result = petrick(&coverage, &primes);

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_simplified_cnf_empty_false_states() {
        // all states are positive (no false states)
        let universe = vec![State(0b00), State(0b01), State(0b10), State(0b11)];
        let positives = universe.clone();

        let cnf = simplified_cnf(&positives, &universe, 0);

        // should return empty CNF
        assert_eq!(cnf.clauses().len(), 0);
    }

    #[test]
    fn test_simplified_cnf_basic() {
        // simple example with some false states
        let universe = vec![State(0b00), State(0b01), State(0b10), State(0b11)];
        let positives = vec![State(0b10), State(0b11)];
        let target = 1;

        let cnf = simplified_cnf(&positives, &universe, target);

        // should generate a CNF that covers the false states
        // and only includes clauses with the target variable
        for clause in &cnf.clauses() {
            assert!(clause.literals().iter().any(|lit| lit.var == target));
        }
    }

    #[test]
    fn test_simplified_cnf_filters_target_var() {
        // test that only clauses with target variable are included
        let universe = vec![
            State(0b000),
            State(0b001),
            State(0b010),
            State(0b011),
            State(0b100),
            State(0b101),
            State(0b110),
            State(0b111),
        ];
        let positives = vec![State(0b111)];
        let target = 2;

        let cnf = simplified_cnf(&positives, &universe, target);

        // all clauses should mention the target variable
        for clause in &cnf.clauses() {
            assert!(
                clause.literals().iter().any(|lit| lit.var == target),
                "Clause should contain target variable {}",
                target
            );
        }
    }
}
