use crate::state::{BitStorage, State16, StateGeneric};

#[derive(Debug, Clone, Copy)]
pub enum Task {
    Add,
    Mul,
    Xor,
    Nand,
    KeepX,
}

impl Task {
    pub fn label(&self) -> &'static str {
        match self {
            Task::Add => "addition",
            Task::Mul => "multiplication",
            Task::Xor => "xor",
            Task::Nand => "nand",
            Task::KeepX => "keepx",
        }
    }

    // Keep backwards compatibility with State16
    pub fn apply(&self, state: &State16) -> State16 {
        match self {
            Task::Add => state.add(),
            Task::Mul => state.mul(),
            Task::Xor => state.xor(),
            Task::Nand => state.nand(),
            Task::KeepX => state.keepx(),
        }
    }

    // Generic version for any BitStorage type
    pub fn apply_generic<B: BitStorage>(&self, state: &StateGeneric<B>) -> StateGeneric<B> {
        match self {
            Task::Add => state.add(),
            Task::Mul => state.mul(),
            Task::Xor => state.xor(),
            Task::Nand => state.nand(),
            Task::KeepX => state.keepx(),
        }
    }
}
