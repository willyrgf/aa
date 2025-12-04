use crate::dataset::Dataset;
use crate::four_bits::Bits4;

pub trait Policy {
    fn predict(&self, input: Bits4) -> Bits4;
    fn complexity(&self) -> u32;
    fn generalisability(&self, dataset: &Dataset) -> u32;
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub mask: u8,  // bits of input we care
    pub value: u8, // required value of those bits
    pub out: Bits4,
}

#[derive(Debug, Clone)]
pub struct RulePolicy {
    pub rules: Vec<Rule>,
    pub default: Option<Bits4>, // when no rule matches
}

impl RulePolicy {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            default: None,
        }
    }

    pub fn with_default(default: Bits4) -> Self {
        Self {
            rules: Vec::new(),
            default: Some(default),
        }
    }

    pub fn add_rule(&mut self, mask: u8, value: u8, out: Bits4) {
        self.rules.push(Rule { mask, value, out });
    }
}

impl Policy for RulePolicy {
    fn predict(&self, input: Bits4) -> Bits4 {
        for rule in &self.rules {
            if (input & rule.mask) == (rule.value & rule.mask) {
                return rule.out;
            }
        }
        self.default.unwrap_or(0)
    }

    fn complexity(&self) -> u32 {
        let mut bits = 0u32;
        for rule in &self.rules {
            bits += rule.mask.count_ones();
        }
        bits + (self.rules.len() as u32)
    }

    fn generalisability(&self, _dataset: &Dataset) -> u32 {
        let mut wildcard_bits = 0u32;
        for rule in &self.rules {
            wildcard_bits += 4 - rule.mask.count_ones()
        }
        wildcard_bits
    }
}
