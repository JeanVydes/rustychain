//! Storage
//!
//! This module provides storage-related pluggable backends.

pub mod memory;
pub mod persistent;

pub use persistent::pgvector::*;
