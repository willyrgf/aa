use crate::state::State;

#[derive(Debug, Clone, Copy)]
pub enum Task {
    Add,
    Mul,
    Xor,
    Nand,
}

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
