//! Composition models
//!
//! This module contains three composition models:
//!  * `MultiClassModel`: combine multiple binary decision models to a single multi-class model
//!  * `MultiTargetModel`: combine multiple univariate models to a single multi-target model
//!  * `Platt`: calibrate a classifier (i.e. SVC) to predicted posterior probabilities
//!
//! `ResidualChain` (forward stagewise additive modeling / L2Boosting) is temporarily
//! orphaned pending a structural rework of its `Fit` impl trait bounds — the same
//! `where F1: Fit + ParamGuard` shape that triggers infinite trait-solver recursion
//! under ndarray 0.17 (see linfa-ensemble's AdaBoost for the same class of issue).
//! Source file `residual_chain.rs` remains in tree for the follow-up PR.
mod multi_class_model;
mod multi_target_model;
pub mod platt_scaling;

pub use multi_class_model::MultiClassModel;
pub use multi_target_model::MultiTargetModel;
pub use platt_scaling::{Platt, PlattError, PlattParams};
