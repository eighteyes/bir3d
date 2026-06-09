//! vs-core — Vector System engine core (orchestration brain; stubbed in the foundation increment).

pub mod fluid;

/// Engine ABI version. Bumped when the WASM↔JS command-descriptor layout changes.
pub fn abi_version() -> u32 { 1 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abi_version_is_stable() { assert_eq!(abi_version(), 1); }
}
