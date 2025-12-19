use std::sync::OnceLock;

static VERBOSITY: OnceLock<u8> = OnceLock::new();

pub fn init(level: u8) {
    let _ = VERBOSITY.set(level);
}

pub fn level() -> u8 {
    *VERBOSITY.get().unwrap_or(&0)
}

pub fn enabled(want: u8) -> bool {
    level() >= want
}

pub fn parse_verbosity<I: IntoIterator<Item = String>>(args: I) -> u8 {
    let mut level = 0u8;

    for arg in args {
        if arg == "--verbose" {
            level = level.saturating_add(1);
            continue;
        }

        // parsing the -v or -vvv
        if let Some(rest) = arg.strip_prefix("-") {
            if !rest.is_empty() && rest.chars().all(|c| c == 'v') {
                let n = rest.len().min(u8::MAX as usize) as u8;
                level = level.saturating_add(n);
            }
        }
    }

    level
}

#[macro_export]
macro_rules! vprintln {
    ($level:expr, $($arg:tt)*) => {
        if $crate::verbosity::enabled($level) {
            eprintln!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! vprint {
    ($level:expr, $($arg:tt)*) => {
        if $crate::verbosity::enabled($level) {
            eprint!($($arg)*);
        }
    };
}
