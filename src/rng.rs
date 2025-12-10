use std::u64;

#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }

        if max.is_power_of_two() {
            return (self.next_u64() as usize) & (max - 1);
        }

        let max_u64 = max as u64;
        let threshold = u64::MAX - (u64::MAX % max_u64);

        loop {
            let x = self.next_u64();
            if x < threshold {
                return (x % max_u64) as usize;
            }
        }
    }
}

#[test]
fn test_next_u64_deterministic() {
    let mut rng = SplitMix64::new(0);
    assert_eq!(rng.next_u64(), 16294208416658607535);
    assert_eq!(rng.next_u64(), 7960286522194355700);
    assert_eq!(rng.next_u64(), 487617019471545679);
}

#[test]
fn test_next_u64_different_seeds() {
    let mut rng1 = SplitMix64::new(1);
    let mut rng2 = SplitMix64::new(2);
    assert_ne!(rng1.next_u64(), rng2.next_u64());
}

#[test]
fn test_next_usize_zero_max() {
    let mut rng = SplitMix64::new(42);
    assert_eq!(rng.next_usize(0), 0);
}

#[test]
fn test_next_usize_power_of_two() {
    let mut rng = SplitMix64::new(12345);

    for _ in 0..10 {
        assert!(rng.next_usize(8) < 8);
        assert!(rng.next_usize(16) < 16);
        assert!(rng.next_usize(256) < 256);
    }
}

#[test]
fn test_next_usize_non_power_of_two() {
    let mut rng = SplitMix64::new(67890);

    for _ in 0..10 {
        assert!(rng.next_usize(10) < 10);
        assert!(rng.next_usize(100) < 100);
    }
}
