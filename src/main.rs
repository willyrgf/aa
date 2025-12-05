use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::time::Instant;

type State = u16;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct Literal {
    var: u8,
    neg: bool,
}

type Clause = Vec<Literal>;
type Cnf = Vec<Clause>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct Term {
    bits: u16,
    mask: u16, // 1 bits are don't-cares
}

#[derive(Default)]
struct Stats {
    trials: usize,
    found: usize,
    perfect: usize,
    tp_sum: f32,
    fp_sum: f32,
    fn_sum: f32,
    weakness_sum: usize,
    length_sum: usize,
    clauses_sum: usize,
    ext_sum: usize,
}

impl Stats {
    fn update(&mut self, policy: Option<Cnf>, universe: &[State], dn: &[State], target: u8) {
        self.trials += 1;
        if let Some(cnf) = policy {
            self.found += 1;
            let (tp, fp, fn_) = accuracy_on_dn(&cnf, universe, dn, target);
            let weak = weakness(&cnf, universe);
            if (tp - 1.0).abs() < f32::EPSILON {
                self.perfect += 1;
            }
            self.tp_sum += tp;
            self.fp_sum += fp;
            self.fn_sum += fn_;
            self.weakness_sum += weak;
            self.length_sum += description_length(&cnf);
            self.clauses_sum += cnf.len();
            self.ext_sum += weak;
        }
    }
}

#[derive(Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max
    }
}

// Sample k distinct elements from dn using a simple Fisher–Yates shuffle driven by the LCG.
fn sample_dk(dn: &[State], k: usize, rng: &mut Lcg) -> Vec<State> {
    let n = dn.len();
    let k = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        indices.swap(i, j);
    }
    indices[..k].iter().map(|&idx| dn[idx]).collect()
}

// Bit layout: bits 0-2 = x (3 bits), 3-5 = y (3 bits), 6-11 = z (6 bits).
fn encode_xy_z(x: u8, y: u8, z: u8) -> State {
    let mut s: u16 = 0;
    s |= ((x & 0b111) as u16) << 0;
    s |= ((y & 0b111) as u16) << 3;
    s |= ((z & 0b111111) as u16) << 6;
    s
}

#[derive(Clone, Copy)]
enum Task {
    Addition,
    Multiplication,
}

impl Task {
    fn label(&self) -> &'static str {
        match self {
            Task::Addition => "addition",
            Task::Multiplication => "multiplication",
        }
    }

    fn operation(&self, x: u8, y: u8) -> u8 {
        match self {
            Task::Addition => (x + y) & 0b111111,
            Task::Multiplication => ((x as u16 * y as u16) & 0b111111) as u8,
        }
    }
}

fn generate_dn(task: Task) -> Vec<State> {
    let mut dn = Vec::new();
    for x in 0u8..8 {
        for y in 0u8..8 {
            let z = task.operation(x, y);
            dn.push(encode_xy_z(x, y, z));
        }
    }
    dn
}

fn popcount16(x: u16) -> u32 {
    x.count_ones()
}

fn combine_terms(a: Term, b: Term) -> Option<Term> {
    if a.mask != b.mask {
        return None;
    }
    let diff = a.bits ^ b.bits;
    if diff.count_ones() != 1 {
        return None;
    }
    if diff & a.mask != 0 {
        return None;
    }
    Some(Term {
        bits: a.bits & !diff,
        mask: a.mask | diff,
    })
}

fn term_covers(term: Term, state: State) -> bool {
    (state & !term.mask) == (term.bits & !term.mask)
}

fn implicant_literal_count(term: Term) -> usize {
    12usize - popcount16(term.mask) as usize
}

fn qm_prime_implicants(false_states: &[State]) -> Vec<Term> {
    let mut groups: Vec<Vec<Term>> = vec![Vec::new(); 13];
    for &s in false_states {
        let ones = popcount16(s) as usize;
        groups[ones].push(Term { bits: s, mask: 0 });
    }

    let mut prime_implicants = Vec::new();
    let mut current = groups;

    loop {
        let mut next: Vec<Vec<Term>> = vec![Vec::new(); 13];
        let mut combined: HashSet<Term> = HashSet::new();
        let mut any_combined = false;

        for i in 0..current.len().saturating_sub(1) {
            for &a in &current[i] {
                for &b in &current[i + 1] {
                    if let Some(c) = combine_terms(a, b) {
                        any_combined = true;
                        combined.insert(a);
                        combined.insert(b);
                        let ones = popcount16(c.bits & !c.mask) as usize;
                        if !next[ones].contains(&c) {
                            next[ones].push(c);
                        }
                    }
                }
            }
        }

        for bucket in &current {
            for &term in bucket {
                if !combined.contains(&term) && !prime_implicants.contains(&term) {
                    prime_implicants.push(term);
                }
            }
        }

        if !any_combined {
            break;
        }
        current = next;
    }

    prime_implicants
}

fn cost_of_combo(combo: &[usize], primes: &[Term]) -> (usize, usize) {
    let lits: usize = combo
        .iter()
        .map(|&i| implicant_literal_count(primes[i]))
        .sum();
    (combo.len(), lits)
}

fn petrick(coverage: &[Vec<usize>], primes: &[Term]) -> Vec<usize> {
    let mut products: Vec<Vec<usize>> = vec![Vec::new()];

    for options in coverage {
        let mut new_products = Vec::new();
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
        .min_by_key(|c| cost_of_combo(c, primes))
        .unwrap_or_default()
}

fn select_implicants(false_states: &[State], primes: &[Term]) -> Vec<Term> {
    let mut remaining: Vec<State> = false_states.to_vec();
    let mut coverage: Vec<Vec<usize>> = Vec::new();
    for &mt in &remaining {
        let options: Vec<usize> = primes
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| if term_covers(p, mt) { Some(i) } else { None })
            .collect();
        coverage.push(options);
    }

    let mut selected: Vec<usize> = Vec::new();

    loop {
        let mut added = false;
        let mut to_remove = Vec::new();
        for (i, opts) in coverage.iter().enumerate() {
            if opts.len() == 1 {
                let only = opts[0];
                if !selected.contains(&only) {
                    selected.push(only);
                }
                to_remove.push(i);
                added = true;
            }
        }

        if added {
            let mut new_remaining = Vec::new();
            let mut new_coverage = Vec::new();
            for (idx, &mt) in remaining.iter().enumerate() {
                let covered = selected
                    .iter()
                    .any(|&pi| term_covers(primes[pi], mt));
                if covered {
                    continue;
                }
                new_remaining.push(mt);
                new_coverage.push(coverage[idx].clone());
            }
            remaining = new_remaining;
            coverage = new_coverage;
        } else {
            break;
        }
    }

    if remaining.is_empty() {
        selected.sort_unstable();
        selected.dedup();
        return selected.into_iter().map(|i| primes[i]).collect();
    }

    let extra = petrick(&coverage, primes);
    for idx in extra {
        if !selected.contains(&idx) {
            selected.push(idx);
        }
    }
    selected.sort_unstable();
    selected.dedup();
    selected.into_iter().map(|i| primes[i]).collect()
}

fn term_to_clause(term: Term) -> Clause {
    let mut clause = Vec::new();
    for var in 0u8..12 {
        if (term.mask >> var) & 1 == 1 {
            continue;
        }
        let bit_is_one = ((term.bits >> var) & 1) != 0;
        clause.push(Literal {
            var,
            neg: bit_is_one,
        });
    }
    clause
}

fn clause_subsumes(a: &Clause, b: &Clause) -> bool {
    a.iter().all(|lit_a| b.contains(lit_a))
}

fn absorb_clauses(cnf: &mut Cnf) {
    if cnf.is_empty() {
        return;
    }
    cnf.sort_by_key(|c| c.len());
    let mut keep: Vec<Clause> = Vec::new();
    'outer: for i in 0..cnf.len() {
        for j in 0..i {
            if clause_subsumes(&keep[j], &cnf[i]) {
                continue 'outer;
            }
        }
        keep.push(cnf[i].clone());
    }
    *cnf = keep;
}

fn simplified_cnf(positives: &[State], universe: &[State], target: u8) -> Cnf {
    let positives_set: HashSet<State> = positives.iter().copied().collect();
    let false_states: Vec<State> = universe
        .iter()
        .copied()
        .filter(|s| !positives_set.contains(s))
        .collect();

    if false_states.is_empty() {
        return Vec::new();
    }

    let primes = qm_prime_implicants(&false_states);
    let implicants = select_implicants(&false_states, &primes);
    let mut cnf: Cnf = implicants.into_iter().map(term_to_clause).collect();
    absorb_clauses(&mut cnf);

    cnf.retain(|clause| clause.iter().any(|lit| lit.var == target));
    cnf
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
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy)]
enum Objective {
    Weakness,
    Simplicity,
}

fn main() {
    let universe: Vec<State> = (0u16..4096).collect();
    let ks = [6usize, 10, 14];
    let trials_per_k = 200;
    let tasks = [Task::Addition, Task::Multiplication];
    let base_seed = 123_456_789;

    println!("Universe size: {}", universe.len());
    println!("Trials per k : {}", trials_per_k);

    for (task_idx, task) in tasks.iter().enumerate() {
        let dn = generate_dn(*task);
        let mut rng = Lcg::new(base_seed + task_idx as u64);
        println!("\n-- Task: {} --", task.label());
        println!("D_n size     : {}", dn.len());

        for &k in &ks {
            let mut w_stats = Stats::default();
            let mut s_stats = Stats::default();

            for _ in 0..trials_per_k {
                let target = rng.next_usize(12) as u8;
                let dk = sample_dk(&dn, k, &mut rng);
                let base_cnf = simplified_cnf(&dk, &universe, target);

                let w_policy = best_first_policy(&base_cnf, &universe, &dk, target, Objective::Weakness, 4, 5000);
                let s_policy =
                    best_first_policy(&base_cnf, &universe, &dk, target, Objective::Simplicity, 4, 5000);

                w_stats.update(w_policy, &universe, &dn, target);
                s_stats.update(s_policy, &universe, &dn, target);
            }

            println!("\n=== |D_k| = {} ===", k);
            print_policy_stats("w-max", &w_stats);
            print_policy_stats("simp-max", &s_stats);
        }
    }
}

fn print_policy_stats(label: &str, stats: &Stats) {
    println!("{}:", label);
    println!(
        "  found   : {}/{} (perfect {} | Rate {:.3})",
        stats.found,
        stats.trials,
        stats.perfect,
        stats.perfect as f32 / stats.trials as f32
    );

    if stats.found == 0 {
        return;
    }

    let denom = stats.found as f32;
    let universe_size = 4096.0;
    let avg_ext = stats.ext_sum as f32 / (denom * universe_size);
    println!(
        "  AvgExt={:.3}  avg weakness: {:.2}  avg length: {:.2}  avg clauses: {:.2}",
        avg_ext,
        stats.weakness_sum as f32 / denom,
        stats.length_sum as f32 / denom,
        stats.clauses_sum as f32 / denom
    );
    println!(
        "  avg TP={:.3} FP={:.3} FN={:.3}",
        stats.tp_sum / denom,
        stats.fp_sum / denom,
        stats.fn_sum / denom
    );
}

fn get_bit(state: State, var: u8) -> bool {
    ((state >> var) & 1) != 0
}

fn eval_literal(lit: &Literal, state: State) -> bool {
    let v = get_bit(state, lit.var);
    if lit.neg { !v } else { v }
}

fn eval_clause(clause: &Clause, state: State) -> bool {
    clause.iter().any(|lit| eval_literal(lit, state))
}

fn eval_cnf(cnf: &Cnf, state: State) -> bool {
    cnf.iter().all(|cl| eval_clause(cl, state))
}

fn weakness(cnf: &Cnf, universe: &[State]) -> usize {
    universe.iter().filter(|&&s| eval_cnf(cnf, s)).count()
}

fn literal_str(lit: &Literal) -> String {
    if lit.neg {
        format!("~x{}", lit.var)
    } else {
        format!("x{}", lit.var)
    }
}

fn clause_length(clause: &Clause) -> usize {
    if clause.is_empty() {
        return 0;
    }
    let mut len = 2; // opening + closing paren
    for (i, lit) in clause.iter().enumerate() {
        len += literal_str(lit).len();
        if i + 1 != clause.len() {
            len += 3; // " | "
        }
    }
    len
}

fn description_length(cnf: &Cnf) -> usize {
    if cnf.is_empty() {
        return 0;
    }
    let mut len = 0;
    for (i, clause) in cnf.iter().enumerate() {
        len += clause_length(clause);
        if i + 1 != cnf.len() {
            len += 3; // " & "
        }
    }
    len
}

fn same_situation(a: State, b: State, target: u8) -> bool {
    let mask: u16 = !(1u16 << target);
    (a & mask) == (b & mask)
}

fn extension(cnf: &Cnf, universe: &[State]) -> Vec<State> {
    universe
        .iter()
        .copied()
        .filter(|s| eval_cnf(cnf, *s))
        .collect()
}

fn reconstruct_decisions(cnf: &Cnf, universe: &[State], situations: &[State], target: u8) -> Vec<State> {
    let ext = extension(cnf, universe);
    let mut result = Vec::new();

    for &sit in situations {
        for &z in &ext {
            if same_situation(z, sit, target) {
                result.push(z);
            }
        }
    }

    result.sort_unstable();
    result.dedup();
    result
}

fn states_equal(a: &[State], b: &[State]) -> bool {
    let mut a_sorted = a.to_vec();
    let mut b_sorted = b.to_vec();
    a_sorted.sort_unstable();
    b_sorted.sort_unstable();
    a_sorted == b_sorted
}

fn is_sufficient(cnf: &Cnf, universe: &[State], decisions: &[State], target: u8) -> bool {
    let recon = reconstruct_decisions(cnf, universe, decisions, target);
    states_equal(&recon, decisions)
}

fn accuracy_on_dn(cnf: &Cnf, universe: &[State], dn: &[State], target: u8) -> (f32, f32, f32) {
    let recon = reconstruct_decisions(cnf, universe, dn, target);

    let recon_set: HashSet<State> = recon.into_iter().collect();
    let dn_set: HashSet<State> = dn.iter().copied().collect();

    let intersection_size = dn_set.intersection(&recon_set).count() as f32;
    let dn_size = dn.len() as f32;
    let recon_size = recon_set.len() as f32;

    let tp = intersection_size / dn_size;
    let fp = if recon_size > 0.0 {
        (recon_size - intersection_size) / dn_size
    } else {
        0.0
    };
    let fn_ = (dn_size - intersection_size) / dn_size;

    (tp, fp, fn_)
}

fn necessary_clauses(cnf: &Cnf, universe: &[State], decisions: &[State], target: u8) -> Vec<usize> {
    let n = cnf.len();
    let mut necessary = Vec::new();
    if n == 0 {
        return necessary;
    }

    for i in 0..n {
        let mut candidate = Vec::with_capacity(n - 1);
        for (j, clause) in cnf.iter().enumerate() {
            if j != i {
                candidate.push(clause.clone());
            }
        }

        if !is_sufficient(&candidate, universe, decisions, target) {
            necessary.push(i);
        }
    }

    if n == 1 && necessary.is_empty() && is_sufficient(cnf, universe, decisions, target) {
        necessary.push(0);
    }

    necessary
}

fn cnf_from_indices(cnf: &Cnf, indices: &[usize]) -> Cnf {
    let mut out = Vec::with_capacity(indices.len());
    for &i in indices {
        out.push(cnf[i].clone());
    }
    out
}

fn best_first_policy(
    full_cnf: &Cnf,
    universe: &[State],
    decisions: &[State],
    target: u8,
    objective: Objective,
    depth_limit: usize,
    time_limit_ms: u128,
) -> Option<Cnf> {
    let start_time = Instant::now();
    let n = full_cnf.len();
    if n == 0 {
        return Some(Vec::new());
    }

    let necessary_idxs = necessary_clauses(full_cnf, universe, decisions, target);

    if necessary_idxs.len() == n {
        return if is_sufficient(full_cnf, universe, decisions, target) {
            Some(full_cnf.clone())
        } else {
            None
        };
    }

    let mut heap = BinaryHeap::new();
    let mut visited: HashSet<Vec<usize>> = HashSet::new();

    let mut start_indices = necessary_idxs.clone();
    start_indices.sort_unstable();
    start_indices.dedup();

    let start_cnf = cnf_from_indices(full_cnf, &start_indices);
    let start_priority = match objective {
        Objective::Weakness => weakness(&start_cnf, universe) as isize,
        Objective::Simplicity => -(description_length(&start_cnf) as isize),
    };

    heap.push(Node {
        indices: start_indices.clone(),
        priority: start_priority,
    });
    visited.insert(start_indices);

    while let Some(node) = heap.pop() {
        if start_time.elapsed().as_millis() > time_limit_ms {
            break;
        }

        let current_cnf = cnf_from_indices(full_cnf, &node.indices);

        if is_sufficient(&current_cnf, universe, decisions, target) {
            return Some(current_cnf);
        }

        if node.indices.len() >= depth_limit {
            continue;
        }

        if node.indices.len() < n {
            for next_idx in 0..n {
                if node.indices.contains(&next_idx) {
                    continue;
                }

                let mut new_indices = node.indices.clone();
                new_indices.push(next_idx);
                new_indices.sort_unstable();
                new_indices.dedup();

                if visited.contains(&new_indices) {
                    continue;
                }

                let new_cnf = cnf_from_indices(full_cnf, &new_indices);
                let pri = match objective {
                    Objective::Weakness => weakness(&new_cnf, universe) as isize,
                    Objective::Simplicity => -(description_length(&new_cnf) as isize),
                };

                visited.insert(new_indices.clone());
                heap.push(Node {
                    indices: new_indices,
                    priority: pri,
                });
            }
        }
    }

    None
}
