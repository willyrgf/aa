use crate::four_bits::{mask4, Bits4};

pub struct Sample {
    pub input: Bits4,
    pub output: Bits4,
}

pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Dataset {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn push(&mut self, input: Bits4, output: Bits4) {
        self.samples.push(Sample { input, output });
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[test]
fn test_new_dataset_is_empty() {
    let dataset = Dataset::new();
    assert_eq!(0, dataset.len());
    assert!(dataset.is_empty());
}

#[test]
fn test_push_samples_and_len() {
    let mut dataset = Dataset::new();

    let input_a = mask4(3);
    let output_a = mask4(12);
    dataset.push(input_a, output_a);
    assert_eq!(1, dataset.len());
    assert!(!dataset.is_empty());

    let input_b = mask4(10);
    let output_b = mask4(5);
    dataset.push(input_b, output_b);
    assert_eq!(2, dataset.len());

    assert_eq!(input_a, dataset.samples[0].input);
    assert_eq!(output_a, dataset.samples[0].output);
    assert_eq!(input_b, dataset.samples[1].input);
    assert_eq!(output_b, dataset.samples[1].output);
}
