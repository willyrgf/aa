use std::fmt;

pub type Bits = u16;

fn input_binary_mask() -> Bits {
    // Input operands use Bits::BITS / 4 bits each
    let input_bits = Bits::BITS / 4;
    (1 << input_bits) - 1
}

fn output_binary_mask() -> Bits {
    // Output uses Bits::BITS / 2 bits
    let output_bits = Bits::BITS / 2;
    (1 << output_bits) - 1
}

fn input_shift_size(input_num: u8) -> u8 {
    (Bits::BITS as u8 / 4) * input_num
}

fn output_shift_size() -> u8 {
    (Bits::BITS / 2) as u8
}

// State a single byte encoding all possible universes 0-255,
// where in z = task(x,y), so the 8 bits are x = 0-1; y = 2-3; z = 4-7
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct State(pub Bits);

// encode returns a State u8: 7-4 = z; 3-2 = y; 0-1 = x
pub fn encode(x: Bits, y: Bits, z: Bits) -> State {
    State(
        ((x & input_binary_mask()) << input_shift_size(0))
            | ((y & input_binary_mask()) << input_shift_size(1))
            | ((z & output_binary_mask()) << output_shift_size()),
    )
}

impl State {
    pub const BITS: u32 = Bits::BITS;

    pub fn decode(&self) -> (Bits, Bits, Bits) {
        (
            (self.0 >> input_shift_size(0) & input_binary_mask()),
            (self.0 >> input_shift_size(1) & input_binary_mask()),
            (self.0 >> output_shift_size() & output_binary_mask()),
        )
    }

    pub fn get_bit(&self, var: u8) -> bool {
        ((self.0 >> var) & 1) != 0
    }

    pub fn ones(&self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn bits(&self) -> Bits {
        self.0
    }

    pub fn add(&self) -> Self {
        let (x, y, _) = self.decode();
        let z = (x + y) & output_binary_mask();
        encode(x, y, z)
    }

    pub fn mul(&self) -> Self {
        let (x, y, _) = self.decode();
        let z = x.wrapping_mul(y) & output_binary_mask();
        encode(x, y, z)
    }

    pub fn xor(&self) -> Self {
        let (x, y, _) = self.decode();
        let z = (x ^ y) & output_binary_mask();
        encode(x, y, z)
    }

    pub fn nand(&self) -> Self {
        let (x, y, _) = self.decode();
        let z = !(x & y) & output_binary_mask();
        encode(x, y, z)
    }

    pub fn keepx(&self) -> Self {
        let (x, y, _) = self.decode();
        let z = x;
        encode(x, y, z)
    }

    // universe return all representable bit patterns.
    // (e.g., u8 is 0..255)
    // would be impossibly big for > u32.
    pub fn universe() -> impl Iterator<Item = State> {
        (Bits::MIN..=Bits::MAX).map(State)
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (x, y, z) = self.decode();
        write!(f, "(x = {}, y = {}, z = {})", x, y, z)
    }
}

// #[test]
// fn test_encode() {
//     assert_eq!(encode(2u8, 2u8, 4u8).0, 74u8);
//     assert_eq!(encode(1u8, 1u8, 8u8).0, 133u8);
//     assert_eq!(encode(3u8, 3u8, 15u8).0, 255u8);
//     assert_eq!(encode(3u8, 3u8, 15u8).decode(), (3u8, 3u8, 15u8));
// }

#[test]
fn test_decode() {
    assert_eq!(encode(0, 0, 0).decode(), (0, 0, 0));
    assert_eq!(encode(1, 1, 1).decode(), (1, 1, 1));
    assert_eq!(encode(2, 2, 4).decode(), (2, 2, 4));
    assert_eq!(encode(1, 1, 8).decode(), (1, 1, 8));
    assert_eq!(encode(3, 3, 15).decode(), (3, 3, 15));
    assert_eq!(encode(0, 3, 0).decode(), (0, 3, 0));
    assert_eq!(encode(1, 0, 15).decode(), (1, 0, 15));
    assert_eq!(encode(3, 1, 7).decode(), (3, 1, 7));
    assert_eq!(encode(2, 3, 10).decode(), (2, 3, 10));
    assert_eq!(encode(3, 2, 12).decode(), (3, 2, 12));
}

#[test]
fn test_display() {
    assert_eq!(State(0).to_string(), "(x = 0, y = 0, z = 0)");
    assert_eq!(encode(0, 0, 0).to_string(), "(x = 0, y = 0, z = 0)");

    assert_eq!(encode(2, 2, 4).to_string(), "(x = 2, y = 2, z = 4)");
    assert_eq!(encode(1, 1, 8).to_string(), "(x = 1, y = 1, z = 8)");

    assert_eq!(encode(3, 3, 15).to_string(), "(x = 3, y = 3, z = 15)");

    assert_eq!(encode(0, 3, 0).to_string(), "(x = 0, y = 3, z = 0)");
    assert_eq!(encode(1, 0, 15).to_string(), "(x = 1, y = 0, z = 15)");
    assert_eq!(encode(3, 1, 7).to_string(), "(x = 3, y = 1, z = 7)");
    assert_eq!(encode(2, 3, 10).to_string(), "(x = 2, y = 3, z = 10)");
}
