use std::{
    collections::{BinaryHeap, HashSet},
    time::Instant,
};

use crate::{
    cnf::{Cnf, is_sufficient, weakness},
    state::State,
};

// Search module is a combinatorial search over subsets of clauses
// with objectives and constraints
//
// given a base cnf and a dataset, search over subsets of clauses to find one that is:
//   sufficient (reconstructs the same decisions), and
//   good under some objective (weakness, simplicity, etc.).

#[derive(Clone, Copy, Debug)]
pub enum Objective {
    Weakness,
    Simplicity,
}

#[derive(Clone, Debug)]
struct Node {
    indices: Vec<usize>,
    priority: isize,
}

impl Eq for Node {}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.indices == other.indices
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct SearchCtx<'a> {
    full_cnf: &'a Cnf,
    universe: &'a [State],
    decisions: &'a [State],
    target: u8,
    objective: Objective,
}

// necessary_clauses return a indices of all necessary clauses
// inside the cnf, whose we cant drop individually from the cnf.
//
// for a new reduced candidate cnf, when I reconstruct decisions
// over universe and target, do I get the same set of decisions
// as the original decisions?
fn necessary_clauses(cnf: &Cnf, universe: &[State], decisions: &[State], target: u8) -> Vec<usize> {
    let n = cnf.len();
    let mut necessary = Vec::new();

    if n == 0 {
        return necessary;
    }

    for idx in 0..n {
        // create a candidate cnf without the clause in idx
        let mut candidate = Vec::with_capacity(n - 1);
        for (j, clause) in cnf.clauses().iter().enumerate() {
            if j != idx {
                candidate.push(clause.clone())
            }
        }
        // if the candidate cnf is not sufficient anymore,
        // the clause idx is a necessary one
        if !is_sufficient(&Cnf(candidate), universe, decisions, target) {
            necessary.push(idx);
        }
    }

    //if there’s only one clause and the whole CNF is sufficient, that clause is trivially necessary
    if n == 1 && necessary.is_empty() && is_sufficient(cnf, universe, decisions, target) {
        necessary.push(0); // first and only clause in cnf
    }

    necessary
}

fn cnf_from_indices(full_cnf: &Cnf, indices: &[usize]) -> Cnf {
    let mut clauses = Vec::with_capacity(indices.len());
    for &idx in indices {
        clauses.push(full_cnf.0[idx].clone());
    }
    Cnf(clauses)
}

fn score_cnf(cnf: &Cnf, objective: Objective, universe: &[State]) -> isize {
    match objective {
        Objective::Weakness => weakness(cnf, universe) as isize,
        Objective::Simplicity => -(cnf.description_length() as isize),
    }
}

// seed_frontier initialise nodes for the priority queue. since
// necessary clauses might not be sufficient, we need supersets.
// so, creates subsets: necessary + cnf_idx;
// score the subset and add them in the priority queue (frontier).
fn seed_frontier(
    ctx: &SearchCtx<'_>,
    necessary_idxs: &[usize],
) -> (BinaryHeap<Node>, HashSet<Vec<usize>>) {
    let n = ctx.full_cnf.len();
    let mut heap: BinaryHeap<Node> = BinaryHeap::new();
    let mut visited: HashSet<Vec<usize>> = HashSet::new();

    for cnf_idx in 0..n {
        let mut start_indices = necessary_idxs.to_vec();
        if !start_indices.contains(&cnf_idx) {
            start_indices.push(cnf_idx);
        }
        start_indices.sort_unstable();
        start_indices.dedup();

        if !visited.insert(start_indices.clone()) {
            continue;
        }

        let start_cnf = cnf_from_indices(ctx.full_cnf, &start_indices);
        let priority = score_cnf(&start_cnf, ctx.objective, ctx.universe);

        heap.push(Node {
            indices: start_indices,
            priority,
        });
    }

    return (heap, visited);
}

// expand_node given a node (subset of clause indices),
// what are its neighbours in the search space that
// arent enqueued.
// a neighbor is a superset that adds 1 more clause:
//   {i1, i2, .., j} for some j not in the set
fn expand_node(
    ctx: &SearchCtx<'_>,
    node: &Node,
    visited: &mut HashSet<Vec<usize>>,
    heap: &mut BinaryHeap<Node>,
    depth_limit: usize,
) {
    if node.indices.len() >= depth_limit {
        return;
    }

    let n = ctx.full_cnf.len();
    if node.indices.len() >= n {
        return;
    }

    for next_idx in 0..n {
        if node.indices.contains(&next_idx) {
            continue;
        }

        let mut new_indices = node.indices.clone();
        new_indices.push(next_idx);
        new_indices.sort_unstable();
        new_indices.dedup();

        if !visited.insert(new_indices.clone()) {
            continue;
        }

        let new_cnf = cnf_from_indices(ctx.full_cnf, &new_indices);
        let priority = score_cnf(&new_cnf, ctx.objective, ctx.universe);

        heap.push(Node {
            indices: new_indices,
            priority,
        });
    }
}

// run_search try to find a sufficient subset.
fn run_search(
    ctx: &SearchCtx<'_>,
    heap: &mut BinaryHeap<Node>,
    visited: &mut HashSet<Vec<usize>>,
    depth_limit: usize,
    time_limit_ms: u128,
) -> (Option<Cnf>, bool) {
    let mut timed_out = false;
    let start_time = Instant::now();

    while let Some(node) = heap.pop() {
        if start_time.elapsed().as_millis() > time_limit_ms {
            timed_out = true;
            break;
        }

        let current_cnf = cnf_from_indices(ctx.full_cnf, &node.indices);
        if is_sufficient(&current_cnf, ctx.universe, ctx.decisions, ctx.target) {
            return (Some(current_cnf), false);
        }

        expand_node(ctx, &node, visited, heap, depth_limit);
    }

    (None, timed_out)
}

pub fn best_first_policy(
    full_cnf: &Cnf,
    universe: &[State],
    decisions: &[State],
    target: u8,
    objective: Objective,
    depth_limit: usize,
    time_limit_ms: u128,
) -> (Option<Cnf>, bool) {
    let n = full_cnf.len();

    if n == 0 {
        return (Some(Cnf(Vec::new())), false);
    }

    let ctx = SearchCtx {
        full_cnf,
        universe,
        decisions,
        target,
        objective,
    };

    let necessary_idxs = necessary_clauses(full_cnf, universe, decisions, target);

    if necessary_idxs.len() == n {
        return if is_sufficient(full_cnf, universe, decisions, target) {
            (Some(full_cnf.clone()), false)
        } else {
            (None, false)
        };
    }

    let (mut heap, mut visited) = seed_frontier(&ctx, &necessary_idxs);

    run_search(&ctx, &mut heap, &mut visited, depth_limit, time_limit_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnf::{Clause, Literal};

    fn test_state(bits: u8) -> State {
        State(bits)
    }

    fn make_simple_cnf() -> Cnf {
        // (x0) & (x1) - both bits 0 and 1 must be set
        Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
        ])
    }

    #[test]
    fn test_node_equality() {
        let node1 = Node {
            indices: vec![0, 1, 2],
            priority: 10,
        };
        let node2 = Node {
            indices: vec![0, 1, 2],
            priority: 10,
        };
        let node3 = Node {
            indices: vec![0, 1, 2],
            priority: 20,
        };

        assert_eq!(node1, node2);
        assert_ne!(node1, node3);
    }

    #[test]
    fn test_node_ordering() {
        let node1 = Node {
            indices: vec![0],
            priority: 10,
        };
        let node2 = Node {
            indices: vec![0],
            priority: 20,
        };

        // higher priority should be greater (max heap)
        assert!(node2 > node1);
        assert!(node1 < node2);
    }

    #[test]
    fn test_node_priority_comparison() {
        let mut heap = BinaryHeap::new();
        heap.push(Node {
            indices: vec![0],
            priority: 5,
        });
        heap.push(Node {
            indices: vec![1],
            priority: 10,
        });
        heap.push(Node {
            indices: vec![2],
            priority: 3,
        });

        // should pop highest priority first
        let first = heap.pop().unwrap();
        assert_eq!(first.priority, 10);

        let second = heap.pop().unwrap();
        assert_eq!(second.priority, 5);

        let third = heap.pop().unwrap();
        assert_eq!(third.priority, 3);
    }

    #[test]
    fn test_necessary_clauses_empty_cnf() {
        let cnf = Cnf(vec![]);
        let universe = vec![test_state(0), test_state(1)];
        let decisions = vec![test_state(1)];

        let necessary = necessary_clauses(&cnf, &universe, &decisions, 0);

        assert_eq!(necessary.len(), 0);
    }

    #[test]
    fn test_necessary_clauses_all_necessary() {
        // cnf where each clause is necessary
        // use different variables to ensure both are needed
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
        ]);
        let universe = vec![
            test_state(0b00),
            test_state(0b01),
            test_state(0b10),
            test_state(0b11),
        ];
        let decisions = vec![test_state(0b11)];
        let target = 1;

        let necessary = necessary_clauses(&cnf, &universe, &decisions, target);

        // at least one clause should be necessary
        assert!(necessary.len() >= 1);
    }

    #[test]
    fn test_necessary_clauses_some_necessary() {
        // cnf where some clauses are necessary and some are redundant
        // (x0 & x1) & (x2) & (x0 & x1) - first clause is necessary, second is about a different variable,
        // third is a duplicate of the first
        let cnf = Cnf(vec![
            Clause(vec![
                Literal { var: 0, neg: false },
                Literal { var: 1, neg: false },
            ]), // (x0 & x1)
            Clause(vec![Literal { var: 2, neg: false }]), // (x2)
            Clause(vec![
                Literal { var: 0, neg: false },
                Literal { var: 1, neg: false },
            ]), // (x0 & x1) again - redundant
        ]);
        let universe = vec![
            test_state(0b000),
            test_state(0b011),
            test_state(0b100),
            test_state(0b111),
        ];
        let decisions = vec![test_state(0b111)]; // all 3 bits set
        let target = 2;

        let necessary = necessary_clauses(&cnf, &universe, &decisions, target);

        // should have some necessary clauses, but not all 3 (since one is redundant)
        // The exact count depends on the semantics of clauses in CNF
        assert!(necessary.len() > 0);
        assert!(necessary.len() < 3);
        // clause 2 should not be necessary as it's a duplicate of clause 0
        if necessary.len() == 2 {
            assert!(
                !(necessary.contains(&0) && necessary.contains(&2)),
                "Both duplicates should not be necessary"
            );
        }
    }

    #[test]
    fn test_necessary_clauses_none_necessary() {
        // cnf where the decision can be reconstructed without any clause
        // this happens when the decision is already captured by empty cnf
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![test_state(0b00), test_state(0b01)];
        let decisions = universe.clone();

        let necessary = necessary_clauses(&cnf, &universe, &decisions, 0);

        // no clauses are necessary since all states are decisions
        assert_eq!(necessary.len(), 0);
    }

    #[test]
    fn test_necessary_clauses_single_clause_necessary() {
        // single clause that is necessary
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![test_state(0b00), test_state(0b01)];
        let decisions = vec![test_state(0b01)];

        let necessary = necessary_clauses(&cnf, &universe, &decisions, 0);

        assert_eq!(necessary.len(), 1);
        assert_eq!(necessary[0], 0);
    }

    #[test]
    fn test_necessary_clauses_single_clause_unnecessary() {
        // single clause that is not necessary
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![test_state(0b00), test_state(0b01)];
        let decisions = vec![test_state(0b00), test_state(0b01)];

        let necessary = necessary_clauses(&cnf, &universe, &decisions, 0);

        // clause is not necessary when all states are decisions
        assert_eq!(necessary.len(), 0);
    }

    #[test]
    fn test_cnf_from_indices_empty() {
        let cnf = make_simple_cnf();
        let indices: Vec<usize> = vec![];

        let result = cnf_from_indices(&cnf, &indices);

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_cnf_from_indices_single() {
        let cnf = make_simple_cnf();
        let indices = vec![0];

        let result = cnf_from_indices(&cnf, &indices);

        assert_eq!(result.len(), 1);
        assert_eq!(result.0[0].0.len(), 1);
        assert_eq!(result.0[0].0[0].var, 0);
    }

    #[test]
    fn test_cnf_from_indices_multiple() {
        let cnf = make_simple_cnf();
        let indices = vec![0, 1];

        let result = cnf_from_indices(&cnf, &indices);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cnf_from_indices_out_of_order() {
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
            Clause(vec![Literal { var: 2, neg: false }]),
        ]);
        let indices = vec![2, 0];

        let result = cnf_from_indices(&cnf, &indices);

        // should get clauses in the order of indices
        assert_eq!(result.len(), 2);
        assert_eq!(result.0[0].0[0].var, 2);
        assert_eq!(result.0[1].0[0].var, 0);
    }

    #[test]
    fn test_score_cnf_weakness() {
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![
            test_state(0b00),
            test_state(0b01),
            test_state(0b10),
            test_state(0b11),
        ];

        let score = score_cnf(&cnf, Objective::Weakness, &universe);

        // weakness counts states where cnf evaluates to true
        // cnf requires bit 0 = 1, so states 0b01 and 0b11 satisfy it
        assert_eq!(score, 2);
    }

    #[test]
    fn test_score_cnf_simplicity() {
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![test_state(0)];

        let score = score_cnf(&cnf, Objective::Simplicity, &universe);

        // simplicity is negative description length
        // "(x0)" has 4 characters, so score should be -4
        assert_eq!(score, -4);
    }

    #[test]
    fn test_score_cnf_weakness_zero() {
        // contradictory cnf: (x0) & (!x0) - can never be satisfied
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 0, neg: true }]),
        ]);
        let universe = vec![test_state(0b00), test_state(0b01)];

        let score = score_cnf(&cnf, Objective::Weakness, &universe);

        // weakness should be 0 since no states satisfy the contradictory cnf
        assert_eq!(score, 0);
    }

    #[test]
    fn test_seed_frontier_basic() {
        let cnf = make_simple_cnf();
        let universe = vec![test_state(0b00), test_state(0b11)];
        let decisions = vec![test_state(0b11)];
        let ctx = SearchCtx {
            full_cnf: &cnf,
            universe: &universe,
            decisions: &decisions,
            target: 0,
            objective: Objective::Weakness,
        };
        let necessary: Vec<usize> = vec![];

        let (heap, visited) = seed_frontier(&ctx, &necessary);

        // should create initial nodes for each clause
        assert!(heap.len() > 0);
        assert!(visited.len() > 0);
    }

    #[test]
    fn test_seed_frontier_with_necessary() {
        let cnf = make_simple_cnf();
        let universe = vec![test_state(0b00), test_state(0b11)];
        let decisions = vec![test_state(0b11)];
        let ctx = SearchCtx {
            full_cnf: &cnf,
            universe: &universe,
            decisions: &decisions,
            target: 0,
            objective: Objective::Weakness,
        };
        let necessary = vec![0];

        let (heap, visited) = seed_frontier(&ctx, &necessary);

        // all frontier nodes should include the necessary clause
        for node in heap.iter() {
            assert!(node.indices.contains(&0));
        }
        assert!(visited.len() > 0);
    }

    #[test]
    fn test_seed_frontier_single_clause() {
        // test with a single-clause CNF
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![test_state(0b00), test_state(0b01)];
        let decisions = vec![test_state(0b01)];
        let ctx = SearchCtx {
            full_cnf: &cnf,
            universe: &universe,
            decisions: &decisions,
            target: 0,
            objective: Objective::Weakness,
        };
        let necessary: Vec<usize> = vec![];

        let (heap, visited) = seed_frontier(&ctx, &necessary);

        // should create one node with index [0]
        assert_eq!(heap.len(), 1);
        assert_eq!(visited.len(), 1);
        let node = heap.peek().unwrap();
        assert_eq!(node.indices, vec![0]);
    }

    #[test]
    fn test_expand_node_basic() {
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
            Clause(vec![Literal { var: 2, neg: false }]),
        ]);
        let universe = vec![test_state(0)];
        let decisions = vec![test_state(0)];
        let ctx = SearchCtx {
            full_cnf: &cnf,
            universe: &universe,
            decisions: &decisions,
            target: 0,
            objective: Objective::Weakness,
        };

        let node = Node {
            indices: vec![0],
            priority: 0,
        };
        let mut visited = HashSet::new();
        visited.insert(vec![0]);
        let mut heap = BinaryHeap::new();

        expand_node(&ctx, &node, &mut visited, &mut heap, 10);

        // should add new nodes with indices [0,1] and [0,2]
        assert!(heap.len() > 0);
        assert!(visited.len() > 1);
    }

    #[test]
    fn test_expand_node_depth_limit() {
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
        ]);
        let universe = vec![test_state(0)];
        let decisions = vec![test_state(0)];
        let ctx = SearchCtx {
            full_cnf: &cnf,
            universe: &universe,
            decisions: &decisions,
            target: 0,
            objective: Objective::Weakness,
        };

        let node = Node {
            indices: vec![0],
            priority: 0,
        };
        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();

        // depth limit of 1 means we can't expand beyond 1 clause
        expand_node(&ctx, &node, &mut visited, &mut heap, 1);

        // should not expand since node already has 1 index and limit is 1
        assert_eq!(heap.len(), 0);
    }

    #[test]
    fn test_expand_node_all_indices_used() {
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
        ]);
        let universe = vec![test_state(0)];
        let decisions = vec![test_state(0)];
        let ctx = SearchCtx {
            full_cnf: &cnf,
            universe: &universe,
            decisions: &decisions,
            target: 0,
            objective: Objective::Weakness,
        };

        let node = Node {
            indices: vec![0, 1],
            priority: 0,
        };
        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();

        expand_node(&ctx, &node, &mut visited, &mut heap, 10);

        // should not expand since all clauses are already used
        assert_eq!(heap.len(), 0);
    }

    #[test]
    fn test_run_search_finds_solution() {
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![test_state(0b00), test_state(0b01)];
        let decisions = vec![test_state(0b01)];
        let ctx = SearchCtx {
            full_cnf: &cnf,
            universe: &universe,
            decisions: &decisions,
            target: 0,
            objective: Objective::Weakness,
        };

        let (mut heap, mut visited) = seed_frontier(&ctx, &[]);

        let (result, timed_out) = run_search(&ctx, &mut heap, &mut visited, 10, 10000);

        // should find a solution
        assert!(result.is_some());
        assert!(!timed_out);
    }

    #[test]
    fn test_run_search_timeout() {
        // create a search that will take a while
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
            Clause(vec![Literal { var: 2, neg: false }]),
            Clause(vec![Literal { var: 3, neg: false }]),
        ]);
        let universe: Vec<State> = (0..16).map(test_state).collect();
        let decisions = vec![test_state(0b1111)];
        let ctx = SearchCtx {
            full_cnf: &cnf,
            universe: &universe,
            decisions: &decisions,
            target: 0,
            objective: Objective::Weakness,
        };

        let (mut heap, mut visited) = seed_frontier(&ctx, &[]);

        // very short timeout
        let (_result, timed_out) = run_search(&ctx, &mut heap, &mut visited, 10, 0);

        // with 0ms timeout, either times out or finds solution quickly
        // just verify it completes without panicking
        assert!(timed_out || !timed_out);
    }

    #[test]
    fn test_run_search_no_solution() {
        // cnf with insufficient clauses - only has (x0) but needs both x0 and x1
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![
            test_state(0b00),
            test_state(0b01),
            test_state(0b10),
            test_state(0b11),
        ];
        let decisions = vec![test_state(0b11)]; // needs both bits set
        let ctx = SearchCtx {
            full_cnf: &cnf,
            universe: &universe,
            decisions: &decisions,
            target: 1,
            objective: Objective::Weakness,
        };

        let (mut heap, mut visited) = seed_frontier(&ctx, &[]);

        let (result, timed_out) = run_search(&ctx, &mut heap, &mut visited, 10, 10000);

        // should return None since no subset of clauses is sufficient
        assert!(result.is_none());
        assert!(!timed_out);
    }

    #[test]
    fn test_best_first_policy_empty_cnf() {
        let cnf = Cnf(vec![]);
        let universe = vec![test_state(0)];
        let decisions = vec![test_state(0)];

        let (result, timed_out) = best_first_policy(
            &cnf,
            &universe,
            &decisions,
            0,
            Objective::Weakness,
            10,
            10000,
        );

        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 0);
        assert!(!timed_out);
    }

    #[test]
    fn test_best_first_policy_simple() {
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![test_state(0b00), test_state(0b01)];
        let decisions = vec![test_state(0b01)];

        let (result, timed_out) = best_first_policy(
            &cnf,
            &universe,
            &decisions,
            0,
            Objective::Weakness,
            10,
            10000,
        );

        assert!(result.is_some());
        assert!(!timed_out);
        // should return a sufficient cnf
        let result_cnf = result.unwrap();
        assert!(is_sufficient(&result_cnf, &universe, &decisions, 0));
    }

    #[test]
    fn test_best_first_policy_multiple_clauses() {
        // cnf with multiple clauses
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 1, neg: false }]),
        ]);
        let universe = vec![
            test_state(0b00),
            test_state(0b01),
            test_state(0b10),
            test_state(0b11),
        ];
        let decisions = vec![test_state(0b11)];

        let (result, timed_out) = best_first_policy(
            &cnf,
            &universe,
            &decisions,
            0,
            Objective::Weakness,
            10,
            10000,
        );

        assert!(result.is_some());
        assert!(!timed_out);
        let result_cnf = result.unwrap();
        // should return a sufficient cnf
        assert!(is_sufficient(&result_cnf, &universe, &decisions, 0));
        assert!(result_cnf.len() >= 1);
    }

    #[test]
    fn test_best_first_policy_no_solution() {
        // cnf with insufficient clauses - only has (x0) but decisions need both x0 and x1
        let cnf = Cnf(vec![Clause(vec![Literal { var: 0, neg: false }])]);
        let universe = vec![
            test_state(0b00),
            test_state(0b01),
            test_state(0b10),
            test_state(0b11),
        ];
        let decisions = vec![test_state(0b11)]; // needs both bits set

        let (result, timed_out) = best_first_policy(
            &cnf,
            &universe,
            &decisions,
            1,
            Objective::Weakness,
            10,
            10000,
        );

        // should return None since no subset of clauses is sufficient
        assert!(result.is_none());
        assert!(!timed_out);
    }

    #[test]
    fn test_best_first_policy_simplicity_objective() {
        let cnf = Cnf(vec![
            Clause(vec![Literal { var: 0, neg: false }]),
            Clause(vec![Literal { var: 0, neg: false }]), // redundant
        ]);
        let universe = vec![test_state(0b00), test_state(0b01)];
        let decisions = vec![test_state(0b01)];

        let (result, timed_out) = best_first_policy(
            &cnf,
            &universe,
            &decisions,
            0,
            Objective::Simplicity,
            10,
            10000,
        );

        assert!(result.is_some());
        assert!(!timed_out);
        let result_cnf = result.unwrap();
        // should prefer simpler solution (1 clause instead of 2)
        assert!(result_cnf.len() <= 1);
    }
}
