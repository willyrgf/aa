use crate::four_bits::{Bits4, mask4};

// linear congruential generator
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    pub fn next_u8(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }

    pub fn next_bits4(&mut self) -> Bits4 {
        mask4(self.next_u8())
    }
}

#[test]
fn test_next_u64_sequence_is_deterministic() {
    let mut rng = Lcg::new(0);
    assert_eq!(0x1, rng.next_u64());
    assert_eq!(0x5851_f42d_4c95_7f2e, rng.next_u64());
    assert_eq!(0xc0b1_8ccf_4e25_2d17, rng.next_u64());
}

#[test]
fn test_next_u32_uses_upper_bits_of_u64() {
    let mut rng = Lcg::new(0);
    assert_eq!(0x0, rng.next_u32());
    assert_eq!(0x5851_f42d, rng.next_u32());
    assert_eq!(0xc0b1_8ccf, rng.next_u32());
}

#[test]
fn test_next_u8_and_next_bits4_are_masked_properly() {
    let mut rng = Lcg::new(0);
    assert_eq!(0, rng.next_u8());
    assert_eq!(45, rng.next_u8());
    assert_eq!(207, rng.next_u8());

    let mut rng = Lcg::new(0);
    assert_eq!(mask4(0), rng.next_bits4());
    assert_eq!(mask4(45), rng.next_bits4());
    assert_eq!(mask4(207), rng.next_bits4());
    assert!(rng.next_bits4() < 16);
}
