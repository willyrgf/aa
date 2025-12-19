use std::fmt;

pub type Bits = u16;

// number of bits actually used from the bits type
// must be even and <= Bits::BITS
// examples: 8 for u8, 12 for u16 (using only 12 of 16 bits), 16 for u16, 24 for u32, etc
pub const BITS_USED: u32 = 16;

fn input_binary_mask() -> Bits {
    // Input operands use BITS_USED / 4 bits each
    let input_bits = BITS_USED / 4;
    (1 << input_bits) - 1
}

fn output_binary_mask() -> Bits {
    // Output uses BITS_USED / 2 bits
    let output_bits = BITS_USED / 2;
    (1 << output_bits) - 1
}

fn input_shift_size(input_num: u8) -> u8 {
    (BITS_USED as u8 / 4) * input_num
}

fn output_shift_size() -> u8 {
    (BITS_USED / 2) as u8
}

// State encoding all possible universes in BITS_USED bits,
// where z = task(x,y). The bits are divided as:
// - x uses bits [0, BITS_USED/4)
// - y uses bits [BITS_USED/4, BITS_USED/2)
// - z uses bits [BITS_USED/2, BITS_USED)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct State(pub Bits);
pub fn encode(x: Bits, y: Bits, z: Bits) -> State {
    State(
        ((x & input_binary_mask()) << input_shift_size(0))
            | ((y & input_binary_mask()) << input_shift_size(1))
            | ((z & output_binary_mask()) << output_shift_size()),
    )
}

impl State {
    pub const BITS: u32 = BITS_USED;

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
    // Only iterates through valid bit patterns (0 to 2^BITS_USED - 1)
    // would be impossibly big for > u32.
    pub fn universe() -> impl Iterator<Item = State> {
        let max_value = if BITS_USED >= Bits::BITS {
            Bits::MAX
        } else {
            (1 << BITS_USED) - 1
        };
        (Bits::MIN..=max_value).map(State)
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (x, y, z) = self.decode();
        write!(f, "(x = {}, y = {}, z = {})", x, y, z)
    }
}

#[test]
fn test_encode_decode_roundtrip() {
    // test that decode(encode(x, y, z)) == (x, y, z) for all valid inputs
    let max_input = (1 << (BITS_USED / 4)) - 1;
    let max_output = (1 << (BITS_USED / 2)) - 1;

    for x in 0..=max_input {
        for y in 0..=max_input {
            for z in 0..=max_output {
                let state = encode(x, y, z);
                assert_eq!(
                    state.decode(),
                    (x, y, z),
                    "round-trip failed for x={}, y={}, z={}",
                    x,
                    y,
                    z
                );
            }
        }
    }
}

#[test]
fn test_boundaries() {
    let max_input = (1 << (BITS_USED / 4)) - 1;
    let max_output = (1 << (BITS_USED / 2)) - 1;

    // test zero
    assert_eq!(encode(0, 0, 0).decode(), (0, 0, 0));

    // test max values
    assert_eq!(
        encode(max_input, max_input, max_output).decode(),
        (max_input, max_input, max_output)
    );

    // test overflow masking
    let overflow_input = max_input + 1;
    let overflow_output = max_output + 1;
    let state = encode(overflow_input, overflow_input, overflow_output);
    let (x, y, z) = state.decode();
    assert_eq!(x, overflow_input & max_input);
    assert_eq!(y, overflow_input & max_input);
    assert_eq!(z, overflow_output & max_output);
}

#[test]
fn test_add_operation() {
    let max_input = (1 << (BITS_USED / 4)) - 1;
    let max_output = (1 << (BITS_USED / 2)) - 1;

    for x in 0..=max_input {
        for y in 0..=max_input {
            let state = encode(x, y, 0);
            let result = state.add();
            let (rx, ry, rz) = result.decode();

            assert_eq!(rx, x, "x should be preserved in add");
            assert_eq!(ry, y, "y should be preserved in add");
            assert_eq!(rz, (x + y) & max_output, "z should be (x + y) masked");
        }
    }
}

#[test]
fn test_mul_operation() {
    let max_input = (1 << (BITS_USED / 4)) - 1;
    let max_output = (1 << (BITS_USED / 2)) - 1;

    for x in 0..=max_input {
        for y in 0..=max_input {
            let state = encode(x, y, 0);
            let result = state.mul();
            let (rx, ry, rz) = result.decode();

            assert_eq!(rx, x, "x should be preserved in mul");
            assert_eq!(ry, y, "y should be preserved in mul");
            assert_eq!(
                rz,
                x.wrapping_mul(y) & max_output,
                "z should be (x * y) masked"
            );
        }
    }
}

#[test]
fn test_xor_operation() {
    let max_input = (1 << (BITS_USED / 4)) - 1;
    let max_output = (1 << (BITS_USED / 2)) - 1;

    for x in 0..=max_input {
        for y in 0..=max_input {
            let state = encode(x, y, 0);
            let result = state.xor();
            let (rx, ry, rz) = result.decode();

            assert_eq!(rx, x, "x should be preserved in xor");
            assert_eq!(ry, y, "y should be preserved in xor");
            assert_eq!(rz, (x ^ y) & max_output, "z should be (x ^ y) masked");
        }
    }
}

#[test]
fn test_nand_operation() {
    let max_input = (1 << (BITS_USED / 4)) - 1;
    let max_output = (1 << (BITS_USED / 2)) - 1;

    for x in 0..=max_input {
        for y in 0..=max_input {
            let state = encode(x, y, 0);
            let result = state.nand();
            let (rx, ry, rz) = result.decode();

            assert_eq!(rx, x, "x should be preserved in nand");
            assert_eq!(ry, y, "y should be preserved in nand");
            assert_eq!(rz, !(x & y) & max_output, "z should be !(x & y) masked");
        }
    }
}

#[test]
fn test_keepx_operation() {
    let max_input = (1 << (BITS_USED / 4)) - 1;

    for x in 0..=max_input {
        for y in 0..=max_input {
            let state = encode(x, y, 0);
            let result = state.keepx();
            let (rx, ry, rz) = result.decode();

            assert_eq!(rx, x, "x should be preserved in keepx");
            assert_eq!(ry, y, "y should be preserved in keepx");
            assert_eq!(rz, x, "z should equal x in keepx");
        }
    }
}

#[test]
fn test_universe_coverage() {
    let expected_count = if BITS_USED >= Bits::BITS {
        (Bits::MAX as usize) + 1
    } else {
        1 << BITS_USED
    };

    let count = State::universe().count();
    assert_eq!(
        count, expected_count,
        "Universe should contain 2^{} states",
        BITS_USED
    );
}

#[test]
fn test_display() {
    // test basic display format
    assert_eq!(State(0).to_string(), "(x = 0, y = 0, z = 0)");
    assert_eq!(encode(0, 0, 0).to_string(), "(x = 0, y = 0, z = 0)");

    // test that display matches decoded values
    let max_input = (1 << (BITS_USED / 4)) - 1;
    let max_output = (1 << (BITS_USED / 2)) - 1;

    let state = encode(max_input, max_input, max_output);
    let (x, y, z) = state.decode();
    assert_eq!(
        state.to_string(),
        format!("(x = {}, y = {}, z = {})", x, y, z)
    );
}
