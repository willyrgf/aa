use std::fmt;

use crate::task::Task;

// State a single byte encoding all possible universes 0-255,
// where in z = task(x,y), so the 8 bits are x = 0-1; y = 2-3; z = 4-7
pub struct State(pub u8);

// encode returns a State u8: 7-4 = z; 3-2 = y; 0-1 = x
pub fn encode(x: u8, y: u8, z: u8) -> State {
    State(((x & 0b11) << 0) | ((y & 0b11) << 2) | ((z & 0b1111) << 4))
}

impl State {
    pub const BITS: u32 = u8::BITS;

    pub fn decode(&self) -> (u8, u8, u8) {
        (
            (self.0 & 0b11),
            ((self.0 >> 2) & 0b11),
            ((self.0 >> 4) & 0b1111),
        )
    }

    pub fn add(&self) -> Self {
        let (x, y, _) = self.decode();
        encode(x, y, (x + y) & 0b1111)
    }

    pub fn mul(&self) -> Self {
        let (x, y, _) = self.decode();
        encode(x, y, ((x as u16 * y as u16) & 0b1111) as u8)
    }

    pub fn xor(&self) -> Self {
        let (x, y, _) = self.decode();
        encode(x, y, (x ^ y) & 0b1111)
    }

    pub fn nand(&self) -> Self {
        let (x, y, _) = self.decode();
        encode(x, y, !(x & y) & 0b1111)
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (x, y, z) = self.decode();
        write!(f, "(x = {}, y = {}, z = {})", x, y, z)
    }
}

// generate_truth_decision_table generate all possible truth states
// for a task based on state size. known as `dn`
pub fn generate_truth_decision_table(task: Task) -> Vec<State> {
    let operand_size = (State::BITS / 2) as u8;
    let mut table = Vec::with_capacity(2usize.pow(operand_size as u32));

    for x in 0u8..operand_size {
        for y in 0u8..operand_size {
            let s = encode(x, y, 0u8);
            table.push(task.apply(&s));
        }
    }

    table
}

#[test]
fn test_encode() {
    assert_eq!(encode(2u8, 2u8, 4u8).0, 74u8);
    assert_eq!(encode(1u8, 1u8, 8u8).0, 133u8);
    assert_eq!(encode(3u8, 3u8, 15u8).0, 255u8);
    assert_eq!(encode(3u8, 3u8, 15u8).decode(), (3u8, 3u8, 15u8));
}

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

#[test]
fn test_generate_truth_decision_table_size() {
    // for 2-bit operands (x and y can be 0-3), we expect 4×4 = 16 entries
    let tasks = [Task::Add, Task::Mul, Task::Xor, Task::Nand];

    for task in tasks {
        let table = generate_truth_decision_table(task);
        assert_eq!(
            table.len(),
            16,
            "{} truth table should have 16 entries",
            task.label()
        );
    }
}

#[test]
fn test_generate_truth_decision_table_add() {
    let table = generate_truth_decision_table(Task::Add);

    // verify each entry has correct x, y inputs and z = (x + y) & 0b1111
    let mut index = 0;
    for x in 0u8..4 {
        for y in 0u8..4 {
            let (got_x, got_y, got_z) = table[index].decode();
            let expected_z = (x + y) & 0b1111;

            assert_eq!(got_x, x, "entry {} should have x = {}", index, x);
            assert_eq!(got_y, y, "entry {} should have y = {}", index, y);
            assert_eq!(
                got_z, expected_z,
                "entry {} (x={}, y={}): expected z={}, got z={}",
                index, x, y, expected_z, got_z
            );

            index += 1;
        }
    }
}

#[test]
fn test_generate_truth_decision_table_mul() {
    let table = generate_truth_decision_table(Task::Mul);

    let mut index = 0;
    for x in 0u8..4 {
        for y in 0u8..4 {
            let (got_x, got_y, got_z) = table[index].decode();
            let expected_z = ((x as u16 * y as u16) & 0b1111) as u8;

            assert_eq!(got_x, x);
            assert_eq!(got_y, y);
            assert_eq!(
                got_z, expected_z,
                "mul entry {} (x={}, y={}): expected z={}, got z={}",
                index, x, y, expected_z, got_z
            );

            index += 1;
        }
    }
}

#[test]
fn test_generate_truth_decision_table_xor() {
    let table = generate_truth_decision_table(Task::Xor);

    let mut index = 0;
    for x in 0u8..4 {
        for y in 0u8..4 {
            let (got_x, got_y, got_z) = table[index].decode();
            let expected_z = (x ^ y) & 0b1111;

            assert_eq!(got_x, x);
            assert_eq!(got_y, y);
            assert_eq!(
                got_z, expected_z,
                "xor entry {} (x={}, y={}): expected z={}, got z={}",
                index, x, y, expected_z, got_z
            );

            index += 1;
        }
    }
}

#[test]
fn test_generate_truth_decision_table_nand() {
    let table = generate_truth_decision_table(Task::Nand);

    let mut index = 0;
    for x in 0u8..4 {
        for y in 0u8..4 {
            let (got_x, got_y, got_z) = table[index].decode();
            let expected_z = !(x & y) & 0b1111;

            assert_eq!(got_x, x);
            assert_eq!(got_y, y);
            assert_eq!(
                got_z, expected_z,
                "nand entry {} (x={}, y={}): expected z={}, got z={}",
                index, x, y, expected_z, got_z
            );

            index += 1;
        }
    }
}
