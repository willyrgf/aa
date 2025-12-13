use crate::state::{Bits, State};
use std::{collections::HashSet, fmt};

#[derive(Debug, Clone)]
pub struct Literal {
    pub var: u8,   // x_var = State.0[var], which bit in the state
    pub neg: bool, // true means !x_var
}

#[derive(Debug, Clone)]
pub struct Clause(pub Vec<Literal>);

// CNF used as boolean classifier to check policies satisfiability (SAT)
#[derive(Debug, Clone)]
pub struct Cnf(pub Vec<Clause>);

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.neg {
            write!(f, "!x{}", self.var)
        } else {
            write!(f, "x{}", self.var)
        }
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "");
        }

        write!(f, "(")?;
        for (i, lit) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " | ")?;
            }
            write!(f, "{}", lit)?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for Cnf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "");
        }

        for (i, clause) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " & ")?;
            }
            write!(f, "{}", clause)?;
        }
        Ok(())
    }
}

impl Literal {
    pub fn eval(&self, s: State) -> bool {
        let x_var = s.get_bit(self.var);
        if self.neg { !x_var } else { x_var }
    }
}

impl Clause {
    // eval at Clause level: Literal OR Literal
    pub fn eval(&self, s: State) -> bool {
        self.0.iter().any(|lit| lit.eval(s))
    }

    pub fn literals(&self) -> Vec<Literal> {
        self.0.clone()
    }
}

impl Cnf {
    pub fn description_length(&self) -> usize {
        self.to_string().len()
    }

    // eval at CNF level: Clause AND Clause
    pub fn eval(&self, s: State) -> bool {
        self.0.iter().all(|clause| clause.eval(s))
    }

    pub fn clauses(&self) -> Vec<Clause> {
        self.0.clone()
    }
}

fn states_eq(a: &[State], b: &[State]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut a_sorted = a.to_vec();
    let mut b_sorted = b.to_vec();
    a_sorted.sort_unstable();
    b_sorted.sort_unstable();

    a_sorted == b_sorted
}

// TODO: find better naming
fn state_eq_except_target(a: State, b: State, target: u8) -> bool {
    // build a mask for target: 0 at target and 1 everywhere else
    let mask: Bits = !(Bits::from(1u8) << target);
    // compare a and b masked: only on the non target bits
    (a.0 & mask) == (b.0 & mask)
}

// extension returns the set of all states in the universe where
// the CNF evaluates true
fn extension(cnf: &Cnf, universe: &[State]) -> Vec<State> {
    universe.iter().copied().filter(|s| cnf.eval(*s)).collect()
}

// TODO: find better naming
fn reconstruct_decision(
    cnf: &Cnf,
    universe: &[State],
    situations: &[State],
    target: u8,
) -> Vec<State> {
    // TODO: find better naming
    let extension = extension(cnf, universe);
    let mut result = Vec::new();

    for &sit in situations {
        for &e in &extension {
            if state_eq_except_target(e, sit, target) {
                result.push(e);
            }
        }
    }

    result.sort_unstable();
    result.dedup();
    result
}

fn is_sufficient(cnf: &Cnf, universe: &[State], decisions: &[State], target: u8) -> bool {
    let reconstructed = reconstruct_decision(cnf, universe, decisions, target);
    states_eq(&reconstructed, decisions)
}

pub fn accuracy_on_decision(
    cnf: &Cnf,
    universe: &[State],
    decision: &[State],
    target: u8,
) -> (f32, f32, f32) {
    let reconstructed = reconstruct_decision(cnf, universe, decision, target);
    let reconstructed_set: HashSet<State> = reconstructed.into_iter().collect();

    let decision_set: HashSet<State> = decision.iter().copied().collect();
    let intersection_set = decision_set.intersection(&reconstructed_set).count() as f32;
    let dn_size = decision_set.len() as f32;
    let recon_size = reconstructed_set.len() as f32;

    // TP (true positive) is the fraction of true decisions that
    // the CNF got right
    let tp = intersection_set / dn_size;

    // FP (false positive) is the fraction of extra decisions that
    // the CNF predicted that are not in the decision truth table
    let fp = if recon_size > 0.0 {
        (recon_size - intersection_set) / dn_size
    } else {
        0.0
    };

    // FN (false negative) is the fraction of true decisions that
    // the CNF misses
    let fn_ = (dn_size - intersection_set) / dn_size;

    (tp, fp, fn_)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state(bits: u8) -> State {
        State(bits as Bits)
    }

    #[test]
    fn test_literal_eval_positive() {
        // x0 (bit 0 = 1) should be true when var=0, neg=false
        let lit = Literal { var: 0, neg: false };
        assert!(lit.eval(test_state(0b00000001)));
        assert!(!lit.eval(test_state(0b00000000)));
    }

    #[test]
    fn test_literal_eval_negative() {
        // !x0 should be true when bit 0 = 0
        let lit = Literal { var: 0, neg: true };
        assert!(lit.eval(test_state(0b00000000)));
        assert!(!lit.eval(test_state(0b00000001)));
    }

    #[test]
    fn test_literal_eval_different_vars() {
        // test different bit positions
        let lit1 = Literal { var: 1, neg: false };
        assert!(lit1.eval(test_state(0b00000010)));
        assert!(!lit1.eval(test_state(0b00000001)));

        let lit2 = Literal { var: 3, neg: false };
        assert!(lit2.eval(test_state(0b00001000)));
        assert!(!lit2.eval(test_state(0b00000111)));
    }

    #[test]
    fn test_clause_eval_empty() {
        // empty clause should be false (no literals to satisfy)
        let clause = Clause(vec![]);
        assert!(!clause.eval(test_state(0b11111111)));
        assert!(!clause.eval(test_state(0b00000000)));
    }

    #[test]
    fn test_clause_eval_single_literal() {
        // (x0) should be true when bit 0 is set
        let clause = Clause(vec![Literal { var: 0, neg: false }]);
        assert!(clause.eval(test_state(0b00000001)));
        assert!(!clause.eval(test_state(0b00000000)));
    }

    #[test]
    fn test_clause_eval_or_logic() {
        // (x0 | x1) should be true if either bit 0 or bit 1 is set
        let clause = Clause(vec![
            Literal { var: 0, neg: false },
            Literal { var: 1, neg: false },
        ]);

        assert!(clause.eval(test_state(0b00000001))); // x0 true
        assert!(clause.eval(test_state(0b00000010))); // x1 true
        assert!(clause.eval(test_state(0b00000011))); // both true
        assert!(!clause.eval(test_state(0b00000000))); // both false
    }

    #[test]
    fn test_clause_eval_mixed_negations() {
        // (x0 | !x1)
        let clause = Clause(vec![
            Literal { var: 0, neg: false },
            Literal { var: 1, neg: true },
        ]);

        assert!(clause.eval(test_state(0b00000001))); // x0=1, x1=0
        assert!(clause.eval(test_state(0b00000000))); // x0=0, x1=0 (!x1 true)
        assert!(clause.eval(test_state(0b00000011))); // x0=1, x1=1
        assert!(!clause.eval(test_state(0b00000010))); // x0=0, x1=1
    }

    #[test]
    fn test_cnf_eval_empty() {
        // empty CNF should be true (no clauses to violate)
        let cnf = Cnf(vec![]);
        assert!(cnf.eval(test_state(0b00000000)));
        assert!(cnf.eval(test_state(0b11111111)));
    }

    #[test]
    fn test_cnf_eval_single_clause() {
        // CNF with single clause (x0)
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        assert!(cnf.eval(test_state(0b00000001)));
        assert!(!cnf.eval(test_state(0b00000000)));
    }

    #[test]
    fn test_cnf_eval_and_logic() {
        // (x0) & (x1) - both must be true
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
        ]);

        assert!(cnf.eval(test_state(0b00000011))); // both true
        assert!(!cnf.eval(test_state(0b00000001))); // only x0 true
        assert!(!cnf.eval(test_state(0b00000010))); // only x1 true
        assert!(!cnf.eval(test_state(0b00000000))); // both false
    }

    #[test]
    fn test_cnf_eval_complex() {
        // (x0 | x1) & (!x2 | x3)
        let cnf = Cnf(vec![
            Clause(vec![
                Literal { var: 0, neg: false },
                Literal { var: 1, neg: false },
            ]),
            Clause(vec![
                Literal { var: 2, neg: true },
                Literal { var: 3, neg: false },
            ]),
        ]);

        // x0=1, x1=0, x2=0, x3=0: (1|0) & (1|0) = true
        assert!(cnf.eval(test_state(0b00000001)));

        // x0=0, x1=1, x2=0, x3=1: (0|1) & (1|1) = true
        assert!(cnf.eval(test_state(0b00001010)));

        // x0=0, x1=0, x2=1, x3=0: (0|0) & (0|0) = false
        assert!(!cnf.eval(test_state(0b00000100)));
    }

    #[test]
    fn test_description_length() {
        // empty CNF displays as "" (0 chars)
        let cnf = Cnf(vec![]);
        assert_eq!(cnf.description_length(), 0);

        // (x0) is 4 chars
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        assert_eq!(cnf.description_length(), 4);

        // (!x0) is 5 chars
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: true }])]);
        assert_eq!(cnf.description_length(), 5);
    }

    #[test]
    fn test_states_eq_identical() {
        let a = vec![test_state(1), test_state(2), test_state(3)];
        let b = vec![test_state(1), test_state(2), test_state(3)];
        assert!(states_eq(&a, &b));
    }

    #[test]
    fn test_states_eq_different_order() {
        let a = vec![test_state(1), test_state(2), test_state(3)];
        let b = vec![test_state(3), test_state(1), test_state(2)];
        assert!(states_eq(&a, &b));
    }

    #[test]
    fn test_states_eq_different_length() {
        let a = vec![test_state(1), test_state(2)];
        let b = vec![test_state(1), test_state(2), test_state(3)];
        assert!(!states_eq(&a, &b));
    }

    #[test]
    fn test_states_eq_different_values() {
        let a = vec![test_state(1), test_state(2)];
        let b = vec![test_state(1), test_state(3)];
        assert!(!states_eq(&a, &b));
    }

    #[test]
    fn test_states_eq_empty() {
        let a: Vec<State> = vec![];
        let b: Vec<State> = vec![];
        assert!(states_eq(&a, &b));
    }

    #[test]
    fn test_state_eq_except_target_same() {
        // states are identical except at target bit
        let a = test_state(0b00000001); // bit 0 = 1
        let b = test_state(0b00000000); // bit 0 = 0
        assert!(state_eq_except_target(a, b, 0));
    }

    #[test]
    fn test_state_eq_except_target_different() {
        // states differ at non-target bit
        let a = test_state(0b00000011); // bits 0,1 = 1
        let b = test_state(0b00000001); // bit 0 = 1, bit 1 = 0
        assert!(!state_eq_except_target(a, b, 0)); // differ at bit 1
    }

    #[test]
    fn test_state_eq_except_target_all_bits() {
        let a = test_state(0b00001111);
        let b = test_state(0b00001110);

        // should be equal except at bit 0
        assert!(state_eq_except_target(a, b, 0));

        // should NOT be equal except at bit 1 (they differ at bit 0 too)
        assert!(!state_eq_except_target(a, b, 1));
    }

    #[test]
    fn test_extension_empty_cnf() {
        // empty CNF is always true, so extension includes all states
        let cnf = Cnf(vec![]);
        let universe = vec![test_state(0), test_state(1), test_state(2)];
        let ext = extension(&cnf, &universe);
        assert_eq!(ext, universe);
    }

    #[test]
    fn test_extension_filters_correctly() {
        // CNF: (x0) - only states with bit 0 set should be in extension
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![
            test_state(0b00000000),
            test_state(0b00000001),
            test_state(0b00000010),
            test_state(0b00000011),
        ];
        let ext = extension(&cnf, &universe);

        // only states with bit 0 = 1
        assert_eq!(ext, vec![test_state(0b00000001), test_state(0b00000011)]);
    }

    #[test]
    fn test_extension_no_matches() {
        // CNF that's impossible to satisfy in given universe
        // (x0) & (!x0)
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 0, neg: true }]),
        ]);
        let universe = vec![test_state(0), test_state(1)];
        let ext = extension(&cnf, &universe);
        assert!(ext.is_empty());
    }

    #[test]
    fn test_reconstruct_decision_simple() {
        // CNF: (x0) - bit 0 must be 1
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![
            test_state(0b00000000), // x0=0, x1=0
            test_state(0b00000001), // x0=1, x1=0
            test_state(0b00000010), // x0=0, x1=1
            test_state(0b00000011), // x0=1, x1=1
        ];

        // situations where we want x1=1 (bit 1 set)
        let situations = vec![
            test_state(0b00000010), // x0=0, x1=1
            test_state(0b00000011), // x0=1, x1=1
        ];

        // target = 1 (we're deciding about bit 1)
        let reconstructed = reconstruct_decision(&cnf, &universe, &situations, 1);

        // CNF allows states where x0=1: [0b01, 0b11]
        // For situation 0b11 (x0=1, x1=1), finds extension states matching on all bits except target:
        // - 0b01 matches (x0=1, differs only on target bit 1)
        // - 0b11 matches (x0=1, differs only on target bit 1)
        // Result: [0b01, 0b11]
        assert_eq!(
            reconstructed,
            vec![test_state(0b00000001), test_state(0b00000011)]
        );
    }

    #[test]
    fn test_is_sufficient_true() {
        // CNF that perfectly captures the decision
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![test_state(0b00000000), test_state(0b00000001)];
        let decisions = vec![test_state(0b00000001)];

        assert!(is_sufficient(&cnf, &universe, &decisions, 0));
    }

    #[test]
    fn test_is_sufficient_false() {
        // CNF that doesn't capture the decision correctly
        let cnf = Cnf(vec![]); // always true
        let universe = vec![test_state(0b00000000), test_state(0b00000001)];
        let decisions = vec![test_state(0b00000001)];

        // reconstruction will include both states, not just decisions
        assert!(!is_sufficient(&cnf, &universe, &decisions, 0));
    }

    #[test]
    fn test_accuracy_on_decision_perfect() {
        // perfect CNF: reconstructs exactly the decision
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![test_state(0b00000000), test_state(0b00000001)];
        let decision = vec![test_state(0b00000001)];

        let (tp, fp, fn_) = accuracy_on_decision(&cnf, &universe, &decision, 0);

        assert_eq!(tp, 1.0); // all true decisions captured
        assert_eq!(fp, 0.0); // no false positives
        assert_eq!(fn_, 0.0); // no false negatives
    }

    #[test]
    fn test_accuracy_on_decision_with_errors() {
        // CNF: always true (reconstructs all states)
        let cnf = Cnf(vec![]);
        let universe = vec![
            test_state(0b00000000),
            test_state(0b00000001),
            test_state(0b00000010),
            test_state(0b00000011),
        ];
        let decision = vec![test_state(0b00000001), test_state(0b00000011)];

        let (tp, fp, fn_) = accuracy_on_decision(&cnf, &universe, &decision, 0);

        // TP: 2/2 = 1.0 (both true decisions are in reconstruction)
        assert_eq!(tp, 1.0);

        // FP: (4-2)/2 = 1.0 (2 extra states reconstructed)
        assert_eq!(fp, 1.0);

        // FN: (2-2)/2 = 0.0 (no true decisions missed)
        assert_eq!(fn_, 0.0);
    }

    #[test]
    fn test_accuracy_on_decision_missing_some() {
        // CNF that only captures one of two decisions
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
        ]);
        let universe = vec![
            test_state(0b00000000),
            test_state(0b00000001),
            test_state(0b00000010),
            test_state(0b00000011),
        ];
        let decision = vec![test_state(0b00000001), test_state(0b00000011)];

        let (tp, _fp, fn_) = accuracy_on_decision(&cnf, &universe, &decision, 0);

        // CNF requires both x0=1 AND x1=1, so only 0b11 satisfies it
        // TP: 1/2 = 0.5 (only one of two decisions captured)
        assert_eq!(tp, 0.5);

        // FN: 1/2 = 0.5 (one decision missed)
        assert_eq!(fn_, 0.5);
    }

    #[test]
    fn test_cnf_display() {
        // empty cnf
        let cnf = Cnf(vec![]);
        assert_eq!(cnf.to_string(), "");

        // single positive literal: x0
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        assert_eq!(cnf.to_string(), "(x0)");

        // single negative literal: !x5
        let cnf = Cnf(vec![Clause(vec![Literal { var: 5, neg: true }])]);
        assert_eq!(cnf.to_string(), "(!x5)");

        // disjunction with two literals: (x0 | x1)
        let cnf = Cnf(vec![Clause(vec![
            Literal { var: 0, neg: false },
            Literal { var: 1, neg: false },
        ])]);
        assert_eq!(cnf.to_string(), "(x0 | x1)");

        // disjunction with three literals: (x0 | x1 | x2)
        let cnf = Cnf(vec![Clause(vec![
            Literal { var: 0, neg: false },
            Literal { var: 1, neg: false },
            Literal { var: 2, neg: false },
        ])]);
        assert_eq!(cnf.to_string(), "(x0 | x1 | x2)");

        // mixed negations: (x0 | !x1 | x2 | !x3)
        let cnf = Cnf(vec![Clause(vec![
            Literal { var: 0, neg: false },
            Literal { var: 1, neg: true },
            Literal { var: 2, neg: false },
            Literal { var: 3, neg: true },
        ])]);
        assert_eq!(cnf.to_string(), "(x0 | !x1 | x2 | !x3)");

        // conjunction of two clauses: (x0) & (x1)
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
        ]);
        assert_eq!(cnf.to_string(), "(x0) & (x1)");

        // three clauses: (x0) & (x1) & (x2)
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
            Clause(vec![Literal { var: 2, neg: false }]),
        ]);
        assert_eq!(cnf.to_string(), "(x0) & (x1) & (x2)");

        // complex formula: (x0 | x1) & (!x2 | x3)
        let cnf = Cnf(vec![
            Clause(vec![
                Literal { var: 0, neg: false },
                Literal { var: 1, neg: false },
            ]),
            Clause(vec![
                Literal { var: 2, neg: true },
                Literal { var: 3, neg: false },
            ]),
        ]);
        assert_eq!(cnf.to_string(), "(x0 | x1) & (!x2 | x3)");

        // more complex: (x0 | !x1 | x2) & (x3) & (!x4 | !x5)
        let cnf = Cnf(vec![
            Clause(vec![
                Literal { var: 0, neg: false },
                Literal { var: 1, neg: true },
                Literal { var: 2, neg: false },
            ]),
            Clause(vec![Literal { var: 3, neg: false }]),
            Clause(vec![
                Literal { var: 4, neg: true },
                Literal { var: 5, neg: true },
            ]),
        ]);
        assert_eq!(cnf.to_string(), "(x0 | !x1 | x2) & (x3) & (!x4 | !x5)");

        // all negated literals: (!x0 | !x1) & (!x2)
        let cnf = Cnf(vec![
            Clause(vec![
                Literal { var: 0, neg: true },
                Literal { var: 1, neg: true },
            ]),
            Clause(vec![Literal { var: 2, neg: true }]),
        ]);
        assert_eq!(cnf.to_string(), "(!x0 | !x1) & (!x2)");

        // clause with many literals: (x0 | x1 | x2 | x3 | x4 | x5)
        let cnf = Cnf(vec![Clause(vec![
            Literal { var: 0, neg: false },
            Literal { var: 1, neg: false },
            Literal { var: 2, neg: false },
            Literal { var: 3, neg: false },
            Literal { var: 4, neg: false },
            Literal { var: 5, neg: false },
        ])]);
        assert_eq!(cnf.to_string(), "(x0 | x1 | x2 | x3 | x4 | x5)");

        // variables with double-digit indices: (x10 | !x15) & (x99)
        let cnf = Cnf(vec![
            Clause(vec![
                Literal {
                    var: 10,
                    neg: false,
                },
                Literal { var: 15, neg: true },
            ]),
            Clause(vec![Literal {
                var: 99,
                neg: false,
            }]),
        ]);
        assert_eq!(cnf.to_string(), "(x10 | !x15) & (x99)");

        // empty clause (edge case)
        let cnf = Cnf(vec![Clause(vec![])]);
        assert_eq!(cnf.to_string(), "");
    }
}
