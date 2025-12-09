use crate::state::State;

#[derive(Debug, Clone, Copy)]
pub enum Task {
    Add,
    Mul,
    Xor,
    Nand,
}

// pub trait TaskOperand: Copy {
//     fn add(self, rhs: Self) -> Self;
//     fn mul(self, rhs: Self) -> Self;
//     fn xor(self, rhs: Self) -> Self;
//     fn nand(self, rhs: Self) -> Self;
// }

impl Task {
    pub fn label(&self) -> &'static str {
        match self {
            Task::Add => "addition",
            Task::Mul => "multiplication",
            Task::Xor => "xor",
            Task::Nand => "nand",
        }
    }

    pub fn apply(&self, state: &State) -> State {
        match self {
            Task::Add => state.add(),
            Task::Mul => state.mul(),
            Task::Xor => state.xor(),
            Task::Nand => state.nand(),
        }
    }
}

// impl TaskOperand for u8 {
//     fn add(self, rhs: Self) -> Self {
//         (self + rhs) & 0b1111
//     }
//
//     fn mul(self, rhs: Self) -> Self {
//         ((self as u16 * rhs as u16) & 0b1111) as u8
//     }
//
//     fn xor(self, rhs: Self) -> Self {
//         (self ^ rhs) & 0b1111
//     }
//
//     fn nand(self, rhs: Self) -> Self {
//         !(self & rhs) & 0b1111
//     }
// }

//
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     fn test_all_pairs_u8(task: Task, expected: &[(u8, u8, u8)]) {
//         for &(x, y, want) in expected {
//             assert_eq!(
//                 task.apply(x, y),
//                 want,
//                 "\n{}({:04b}, {:04b}) should be {:04b} ({}) but got {:04b} ({})",
//                 task.label(),
//                 x,
//                 y,
//                 want,
//                 want,
//                 task.apply(x, y),
//                 task.apply(x, y)
//             );
//         }
//
//         // all operations must stay inside 4 bits by construction
//         for x in 0..16u8 {
//             for y in 0..16u8 {
//                 let got = task.apply(x, y);
//                 assert!(
//                     got < 16,
//                     "overflow in {}({}, {}) = {}",
//                     task.label(),
//                     x,
//                     y,
//                     got
//                 );
//             }
//         }
//     }
//
//     #[test]
//     fn add_u8() {
//         let cases = &[
//             (0, 0, 0),
//             (1, 1, 2),
//             (7, 8, 15),   // 0111 + 1000 = 1111
//             (10, 11, 5),  // 1010 + 1011 = 10101 → & 0b1111 = 0101
//             (15, 1, 0),   // 1111 + 0001 = 10000 → 0000
//             (15, 15, 14), // 1111 + 1111 = 11110 → 1110
//         ];
//         test_all_pairs_u8(Task::Add, cases);
//     }
//
//     #[test]
//     fn mul_u8() {
//         let cases = &[
//             (0, 5, 0),
//             (1, 1, 1),
//             (2, 3, 6),
//             (3, 5, 15),  // 0011 * 0101 = 1111
//             (4, 4, 0),   // 0100 * 0100 = 10000 → 0000
//             (5, 7, 3),   // 0101 * 0111 = 100011 → 0011
//             (15, 15, 1), // 1111 * 1111 = 11100001 → 0001
//         ];
//         test_all_pairs_u8(Task::Mul, cases);
//     }
//
//     #[test]
//     fn xor_u8() {
//         let cases = &[
//             (0b0000, 0b0000, 0b0000),
//             (0b0001, 0b0001, 0b0000),
//             (0b0011, 0b0101, 0b0110),
//             (0b1010, 0b1111, 0b0101),
//             (0b1111, 0b1111, 0b0000),
//         ];
//         test_all_pairs_u8(Task::Xor, cases);
//     }
//
//     #[test]
//     fn nand_u8() {
//         let b_cases = &[
//             (0b0000, 0b0000, 0b1111),             // ~(0 & 0) = ~0  = 1111
//             (0b0000, 0b1111, 0b1111),             // ~(0 & 15) = ~0 = 1111
//             (0b1111, 0b1111, 0b0000),             // ~(15 & 15) = ~15 = 0000 (in 4 bits)
//             (0b1010, 0b1100, 0b0111),             // &(1010 & 1100)=1000 → ~1000 = 0111
//             (0b0011, 0b0101, !(0b0001) & 0b1111), // → 0b1110
//         ];
//         for &(x, y, want) in b_cases {
//             assert_eq!(Task::Nand.apply(x, y), want);
//         }
//
//         let u8_cases = &[
//             (0, 0, 15),
//             (0, 15, 15),
//             (15, 15, 0),
//             (5, 10, !(5 & 10) & 15), // 0101 & 1010 = 0000 → 1111
//             (7, 7, !(7 & 7) & 15),   // 0111 & 0111 = 0111 → ~0111 = 1000
//             (3, 5, !(3 & 5) & 15),   // 0011 & 0101 = 0001 → ~0001 = 1110
//         ];
//         for &(x, y, want) in u8_cases {
//             assert_eq!(Task::Nand.apply(x, y), want);
//         }
//
//         // brute-force the rest
//         for x in 0..16u8 {
//             for y in 0..16u8 {
//                 let got = Task::Nand.apply(x, y);
//                 let manual = !(x & y) & 0b1111;
//                 assert_eq!(got, manual, "nand({:04b}, {:04b}) failed", x, y);
//             }
//         }
//     }
//
//     #[test]
//     fn label_works() {
//         assert_eq!(Task::Add.label(), "addition");
//         assert_eq!(Task::Mul.label(), "multiplication");
//         assert_eq!(Task::Xor.label(), "xor");
//         assert_eq!(Task::Nand.label(), "nand");
//     }
// }
