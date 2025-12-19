use std::fmt;

/// Trait for types that can be used as bit storage in State
pub trait BitStorage:
    Copy
    + Clone
    + Eq
    + Ord
    + std::hash::Hash
    + fmt::Debug
    + fmt::Display
    + From<u8>
    + 'static
{
    /// Total number of bits in the storage type (e.g., 8 for u8, 16 for u16)
    const TOTAL_BITS: u32;

    /// Number of bits actually used (must be even and <= TOTAL_BITS)
    const BITS_USED: u32;

    /// Minimum value (typically 0)
    const MIN: Self;

    /// Maximum value for the storage type
    const MAX: Self;

    /// Count the number of set bits
    fn count_ones(self) -> u32;

    /// Left shift operation
    fn shl(self, rhs: u32) -> Self;

    /// Right shift operation
    fn shr(self, rhs: u32) -> Self;

    /// Bitwise AND
    fn bitand(self, rhs: Self) -> Self;

    /// Bitwise OR
    fn bitor(self, rhs: Self) -> Self;

    /// Bitwise XOR
    fn bitxor(self, rhs: Self) -> Self;

    /// Bitwise NOT
    fn bitnot(self) -> Self;

    /// Addition (wrapping)
    fn add(self, rhs: Self) -> Self;

    /// Multiplication (wrapping)
    fn mul(self, rhs: Self) -> Self;

    /// Checked conversion from usize (for iteration)
    fn from_usize(value: usize) -> Option<Self>;

    /// Convert to usize (for iteration)
    fn to_usize(self) -> usize;

    /// Create a value where only the nth bit is set
    fn from_bit_position(n: u8) -> Self;
}

// Implement BitStorage for u8
impl BitStorage for u8 {
    const TOTAL_BITS: u32 = 8;
    const BITS_USED: u32 = 8;
    const MIN: Self = u8::MIN;
    const MAX: Self = u8::MAX;

    #[inline]
    fn count_ones(self) -> u32 {
        self.count_ones()
    }

    #[inline]
    fn shl(self, rhs: u32) -> Self {
        self.wrapping_shl(rhs)
    }

    #[inline]
    fn shr(self, rhs: u32) -> Self {
        self.wrapping_shr(rhs)
    }

    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    #[inline]
    fn bitnot(self) -> Self {
        !self
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }

    #[inline]
    fn from_usize(value: usize) -> Option<Self> {
        u8::try_from(value).ok()
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn from_bit_position(n: u8) -> Self {
        1u8 << n
    }
}

/// 12-bit unsigned integer type (stored in u16, masked to 12 bits)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct U12(u16);

impl U12 {
    const MASK: u16 = 0x0FFF; // 12 bits: 0000_1111_1111_1111

    /// Create a new U12, masking to 12 bits
    #[inline]
    pub fn new(value: u16) -> Self {
        U12(value & Self::MASK)
    }

    /// Get the inner u16 value (always <= 0x0FFF)
    #[inline]
    pub fn get(self) -> u16 {
        self.0
    }
}

impl From<u8> for U12 {
    #[inline]
    fn from(value: u8) -> Self {
        U12(value as u16)
    }
}

impl fmt::Display for U12 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl BitStorage for U12 {
    const TOTAL_BITS: u32 = 16; // stored in u16
    const BITS_USED: u32 = 12; // only use 12 bits
    const MIN: Self = U12(0);
    const MAX: Self = U12(Self::MASK);

    #[inline]
    fn count_ones(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    fn shl(self, rhs: u32) -> Self {
        U12::new(self.0 << rhs)
    }

    #[inline]
    fn shr(self, rhs: u32) -> Self {
        U12(self.0 >> rhs)
    }

    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        U12(self.0 & rhs.0)
    }

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        U12(self.0 | rhs.0)
    }

    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        U12::new(self.0 ^ rhs.0)
    }

    #[inline]
    fn bitnot(self) -> Self {
        U12::new(!self.0)
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        U12::new(self.0.wrapping_add(rhs.0))
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        U12::new(self.0.wrapping_mul(rhs.0))
    }

    #[inline]
    fn from_usize(value: usize) -> Option<Self> {
        if value <= Self::MASK as usize {
            Some(U12(value as u16))
        } else {
            None
        }
    }

    #[inline]
    fn to_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn from_bit_position(n: u8) -> Self {
        U12(1u16 << n)
    }
}

impl BitStorage for u16 {
    const TOTAL_BITS: u32 = 16;
    const BITS_USED: u32 = 16;
    const MIN: Self = u16::MIN;
    const MAX: Self = u16::MAX;

    #[inline]
    fn count_ones(self) -> u32 {
        self.count_ones()
    }

    #[inline]
    fn shl(self, rhs: u32) -> Self {
        self.wrapping_shl(rhs)
    }

    #[inline]
    fn shr(self, rhs: u32) -> Self {
        self.wrapping_shr(rhs)
    }

    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    #[inline]
    fn bitnot(self) -> Self {
        !self
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }

    #[inline]
    fn from_usize(value: usize) -> Option<Self> {
        u16::try_from(value).ok()
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn from_bit_position(n: u8) -> Self {
        1u16 << n
    }
}

pub type Bits = u16;

// Type aliases for different bit widths
pub type State8 = StateGeneric<u8>;
pub type State12 = StateGeneric<U12>;
pub type State16 = StateGeneric<u16>;
pub type State = State16;

#[inline]
fn input_binary_mask<B: BitStorage>() -> B {
    // Input operands use BITS_USED / 4 bits each
    let input_bits = B::BITS_USED / 4;
    // Create mask: (1 << input_bits) - 1
    let all_ones = B::MIN.bitnot();
    if input_bits >= B::BITS_USED {
        all_ones
    } else {
        all_ones.shr(B::BITS_USED - input_bits)
    }
}

#[inline]
fn output_binary_mask<B: BitStorage>() -> B {
    // Output uses BITS_USED / 2 bits
    let output_bits = B::BITS_USED / 2;
    // Create mask: (1 << output_bits) - 1
    let all_ones = B::MIN.bitnot();
    if output_bits >= B::BITS_USED {
        all_ones
    } else {
        all_ones.shr(B::BITS_USED - output_bits)
    }
}

#[inline]
fn input_shift_size<B: BitStorage>(input_num: u8) -> u32 {
    (B::BITS_USED / 4) * (input_num as u32)
}

#[inline]
fn output_shift_size<B: BitStorage>() -> u32 {
    B::BITS_USED / 2
}

// State encoding all possible universes in BITS_USED bits,
// where z = task(x,y). The bits are divided as:
// - x uses bits [0, BITS_USED/4)
// - y uses bits [BITS_USED/4, BITS_USED/2)
// - z uses bits [BITS_USED/2, BITS_USED)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct StateGeneric<B: BitStorage>(pub B);

pub fn encode<B: BitStorage>(x: B, y: B, z: B) -> StateGeneric<B> {
    let x_masked = x.bitand(input_binary_mask());
    let y_masked = y.bitand(input_binary_mask());
    let z_masked = z.bitand(output_binary_mask());

    let x_shifted = x_masked.shl(input_shift_size::<B>(0));
    let y_shifted = y_masked.shl(input_shift_size::<B>(1));
    let z_shifted = z_masked.shl(output_shift_size::<B>());

    StateGeneric(x_shifted.bitor(y_shifted).bitor(z_shifted))
}

impl<B: BitStorage> StateGeneric<B> {
    pub const BITS: u32 = B::BITS_USED;

    pub fn decode(&self) -> (B, B, B) {
        let x = self.0.shr(input_shift_size::<B>(0)).bitand(input_binary_mask());
        let y = self.0.shr(input_shift_size::<B>(1)).bitand(input_binary_mask());
        let z = self.0.shr(output_shift_size::<B>()).bitand(output_binary_mask());
        (x, y, z)
    }

    pub fn get_bit(&self, var: u8) -> bool {
        let one = B::from_bit_position(0);
        self.0.shr(var as u32).bitand(one) != B::MIN
    }

    pub fn ones(&self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn bits(&self) -> B {
        self.0
    }

    pub fn add(&self) -> Self {
        let (x, y, _) = self.decode();
        let z = x.add(y).bitand(output_binary_mask());
        encode(x, y, z)
    }

    pub fn mul(&self) -> Self {
        let (x, y, _) = self.decode();
        let z = x.mul(y).bitand(output_binary_mask());
        encode(x, y, z)
    }

    pub fn xor(&self) -> Self {
        let (x, y, _) = self.decode();
        let z = x.bitxor(y).bitand(output_binary_mask());
        encode(x, y, z)
    }

    pub fn nand(&self) -> Self {
        let (x, y, _) = self.decode();
        let z = x.bitand(y).bitnot().bitand(output_binary_mask());
        encode(x, y, z)
    }

    pub fn keepx(&self) -> Self {
        let (x, y, _) = self.decode();
        encode(x, y, x)
    }

    // universe return all representable bit patterns.
    // Only iterates through valid bit patterns (0 to 2^BITS_USED - 1)
    // would be impossibly big for > u32.
    pub fn universe() -> impl Iterator<Item = StateGeneric<B>> {
        let max_value = if B::BITS_USED >= B::TOTAL_BITS {
            B::MAX
        } else {
            B::MIN.bitnot()
        };
        StateIterator {
            current: 0,
            end: max_value.to_usize(),
            _phantom: std::marker::PhantomData,
        }
    }
}

// Custom iterator for State::universe()
struct StateIterator<B: BitStorage> {
    current: usize,
    end: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: BitStorage> Iterator for StateIterator<B> {
    type Item = StateGeneric<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current > self.end {
            return None;
        }
        let value = B::from_usize(self.current)?;
        self.current += 1;
        Some(StateGeneric(value))
    }
}

impl<B: BitStorage> fmt::Display for StateGeneric<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (x, y, z) = self.decode();
        write!(f, "(x = {}, y = {}, z = {})", x, y, z)
    }
}

impl<B: BitStorage> From<B> for StateGeneric<B> {
    fn from(value: B) -> Self {
        StateGeneric(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{encode, BitStorage, StateGeneric, U12};

    macro_rules! for_all_bit_widths {
        ($func:ident) => {{
            $func::<u8>();
            $func::<U12>();
            $func::<u16>();
        }};
    }

    fn max_input<B: BitStorage>() -> usize {
        (1usize << (B::BITS_USED / 4)) - 1
    }

    fn max_output<B: BitStorage>() -> usize {
        (1usize << (B::BITS_USED / 2)) - 1
    }

    fn test_encode_decode_roundtrip_for<B: BitStorage>() {
        let max_input = max_input::<B>();
        let max_output = max_output::<B>();

        for x in 0..=max_input {
            for y in 0..=max_input {
                for z in 0..=max_output {
                    let state = encode::<B>(
                        B::from_usize(x).unwrap(),
                        B::from_usize(y).unwrap(),
                        B::from_usize(z).unwrap(),
                    );
                    let (rx, ry, rz) = state.decode();
                    assert_eq!(
                        (rx.to_usize(), ry.to_usize(), rz.to_usize()),
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

    fn test_boundaries_for<B: BitStorage>() {
        let max_input = max_input::<B>();
        let max_output = max_output::<B>();

        let zero = B::from_usize(0).unwrap();
        assert_eq!(encode::<B>(zero, zero, zero).decode(), (zero, zero, zero));

        let state = encode::<B>(
            B::from_usize(max_input).unwrap(),
            B::from_usize(max_input).unwrap(),
            B::from_usize(max_output).unwrap(),
        );
        let (x, y, z) = state.decode();
        assert_eq!(
            (x.to_usize(), y.to_usize(), z.to_usize()),
            (max_input, max_input, max_output)
        );

        let overflow_input = max_input + 1;
        let overflow_output = max_output + 1;
        let state = encode::<B>(
            B::from_usize(overflow_input).unwrap(),
            B::from_usize(overflow_input).unwrap(),
            B::from_usize(overflow_output).unwrap(),
        );
        let (x, y, z) = state.decode();
        assert_eq!(x.to_usize(), overflow_input & max_input);
        assert_eq!(y.to_usize(), overflow_input & max_input);
        assert_eq!(z.to_usize(), overflow_output & max_output);
    }

    fn test_add_operation_for<B: BitStorage>() {
        let max_input = max_input::<B>();
        let max_output = max_output::<B>();

        for x in 0..=max_input {
            for y in 0..=max_input {
                let state = encode::<B>(
                    B::from_usize(x).unwrap(),
                    B::from_usize(y).unwrap(),
                    B::from_usize(0).unwrap(),
                );
                let result = state.add();
                let (rx, ry, rz) = result.decode();

                assert_eq!(rx.to_usize(), x, "x should be preserved in add");
                assert_eq!(ry.to_usize(), y, "y should be preserved in add");
                assert_eq!(
                    rz.to_usize(),
                    (x + y) & max_output,
                    "z should be (x + y) masked"
                );
            }
        }
    }

    fn test_mul_operation_for<B: BitStorage>() {
        let max_input = max_input::<B>();
        let max_output = max_output::<B>();

        for x in 0..=max_input {
            for y in 0..=max_input {
                let state = encode::<B>(
                    B::from_usize(x).unwrap(),
                    B::from_usize(y).unwrap(),
                    B::from_usize(0).unwrap(),
                );
                let result = state.mul();
                let (rx, ry, rz) = result.decode();

                assert_eq!(rx.to_usize(), x, "x should be preserved in mul");
                assert_eq!(ry.to_usize(), y, "y should be preserved in mul");
                assert_eq!(
                    rz.to_usize(),
                    x.wrapping_mul(y) & max_output,
                    "z should be (x * y) masked"
                );
            }
        }
    }

    fn test_xor_operation_for<B: BitStorage>() {
        let max_input = max_input::<B>();
        let max_output = max_output::<B>();

        for x in 0..=max_input {
            for y in 0..=max_input {
                let state = encode::<B>(
                    B::from_usize(x).unwrap(),
                    B::from_usize(y).unwrap(),
                    B::from_usize(0).unwrap(),
                );
                let result = state.xor();
                let (rx, ry, rz) = result.decode();

                assert_eq!(rx.to_usize(), x, "x should be preserved in xor");
                assert_eq!(ry.to_usize(), y, "y should be preserved in xor");
                assert_eq!(
                    rz.to_usize(),
                    (x ^ y) & max_output,
                    "z should be (x ^ y) masked"
                );
            }
        }
    }

    fn test_nand_operation_for<B: BitStorage>() {
        let max_input = max_input::<B>();
        let max_output = max_output::<B>();

        for x in 0..=max_input {
            for y in 0..=max_input {
                let state = encode::<B>(
                    B::from_usize(x).unwrap(),
                    B::from_usize(y).unwrap(),
                    B::from_usize(0).unwrap(),
                );
                let result = state.nand();
                let (rx, ry, rz) = result.decode();

                assert_eq!(rx.to_usize(), x, "x should be preserved in nand");
                assert_eq!(ry.to_usize(), y, "y should be preserved in nand");
                assert_eq!(
                    rz.to_usize(),
                    (!(x & y)) & max_output,
                    "z should be !(x & y) masked"
                );
            }
        }
    }

    fn test_keepx_operation_for<B: BitStorage>() {
        let max_input = max_input::<B>();

        for x in 0..=max_input {
            for y in 0..=max_input {
                let state = encode::<B>(
                    B::from_usize(x).unwrap(),
                    B::from_usize(y).unwrap(),
                    B::from_usize(0).unwrap(),
                );
                let result = state.keepx();
                let (rx, ry, rz) = result.decode();

                assert_eq!(rx.to_usize(), x, "x should be preserved in keepx");
                assert_eq!(ry.to_usize(), y, "y should be preserved in keepx");
                assert_eq!(rz.to_usize(), x, "z should equal x in keepx");
            }
        }
    }

    fn test_universe_coverage_for<B: BitStorage>() {
        let expected_count = if B::BITS_USED >= B::TOTAL_BITS {
            B::MAX.to_usize() + 1
        } else {
            1usize << B::BITS_USED
        };

        let count = StateGeneric::<B>::universe().count();
        assert_eq!(
            count, expected_count,
            "Universe should contain 2^{} states",
            B::BITS_USED
        );
    }

    fn test_display_for<B: BitStorage>() {
        let zero_state = StateGeneric::<B>(B::MIN);
        assert_eq!(zero_state.to_string(), "(x = 0, y = 0, z = 0)");

        let max_input = max_input::<B>();
        let max_output = max_output::<B>();
        let state = encode::<B>(
            B::from_usize(max_input).unwrap(),
            B::from_usize(max_input).unwrap(),
            B::from_usize(max_output).unwrap(),
        );
        let (x, y, z) = state.decode();
        assert_eq!(
            state.to_string(),
            format!("(x = {}, y = {}, z = {})", x, y, z)
        );
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        for_all_bit_widths!(test_encode_decode_roundtrip_for);
    }

    #[test]
    fn test_boundaries() {
        for_all_bit_widths!(test_boundaries_for);
    }

    #[test]
    fn test_add_operation() {
        for_all_bit_widths!(test_add_operation_for);
    }

    #[test]
    fn test_mul_operation() {
        for_all_bit_widths!(test_mul_operation_for);
    }

    #[test]
    fn test_xor_operation() {
        for_all_bit_widths!(test_xor_operation_for);
    }

    #[test]
    fn test_nand_operation() {
        for_all_bit_widths!(test_nand_operation_for);
    }

    #[test]
    fn test_keepx_operation() {
        for_all_bit_widths!(test_keepx_operation_for);
    }

    #[test]
    fn test_universe_coverage() {
        for_all_bit_widths!(test_universe_coverage_for);
    }

    #[test]
    fn test_display() {
        for_all_bit_widths!(test_display_for);
    }
}
