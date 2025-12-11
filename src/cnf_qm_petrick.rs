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

        if combined_set.is_empty() {
            break;
        }

        current = next;
    }

    prime_implicants
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

        assert!(term_covers(term, state));
    }

    #[test]
    fn test_term_covers_with_dont_care() {
        // term 10-0 (bit 1 is dont-care)
        let term = Term {
            bits: 0b1000,
            mask: 0b0010,
        };

        // should cover both 1010 and 1000
        assert!(term_covers(term.clone(), State(0b1010)));
        assert!(term_covers(term.clone(), State(0b1000)));

        // should not cover 0010 or 0000
        assert!(!term_covers(term.clone(), State(0b0010)));
        assert!(!term_covers(term, State(0b0000)));
    }

    #[test]
    fn test_term_covers_not_matching() {
        let term = Term {
            bits: 0b1010,
            mask: 0b0000,
        };
        let state = State(0b1000);

        assert!(!term_covers(term, state));
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
}
