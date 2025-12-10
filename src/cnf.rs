use std::{collections::HashSet, fmt};

use crate::state::{Bits, State, encode};

#[derive(Debug, Clone)]
pub struct Literal {
    pub var: u8,   // x_var = State.0[var], which bit in the state
    pub neg: bool, // true means !x_var
}

#[derive(Debug, Clone)]
pub struct Clause(pub Vec<Literal>);

#[derive(Debug, Clone)]
pub struct Cnf(pub Vec<Clause>);

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        if self.neg {
            write!(f, "!x{}", self.var)
        } else {
            write!(f, "x{}", self.var)
        }
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            // for visibility, might rollback to empty
            return write!(f, "()");
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            // for visibility, might rollback to empty
            return write!(f, "true");
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
}

impl Cnf {
    pub fn description_length(&self) -> usize {
        self.to_string().len()
    }

    // eval at CNF level: Clause AND Clause
    pub fn eval(&self, s: State) -> bool {
        self.0.iter().all(|clause| clause.eval(s))
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
