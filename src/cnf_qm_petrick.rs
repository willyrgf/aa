use std::collections::HashSet;

use crate::state::{Bits, State};

// QM (Quine–McCluskey)
// combine two implicants that differ in exactly one care bit into a more general implicant where that bit becomes dont-care

// Term is a specific state pattern and its mask
// bits = 0b1010, mask = 0b0000 means:
//  bit3=1; bit2=0; bit1=1; bit0=0 (no mask)
// bits = 0b1010, mask = 0b0010 means:
//  bit3=1; bit2=0; bit1=dont-care; bit0=0 (bit1 masked out)
#[derive(Clone, Hash, PartialEq, Eq)]
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
pub fn term_covers(term: Term, state: State) -> bool {
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

        if !combined_set.is_empty() {
            break;
        }

        current = next;
    }

    prime_implicants
}
