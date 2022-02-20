//! This library contains the core functionality of the `ask-bayes` crate.
#![warn(
    clippy::all,
    clippy::restriction,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    rust_2018_idioms,
    missing_debug_implementations,
    missing_docs
)]
#![allow(clippy::module_inception)]
#![allow(clippy::implicit_return)]
#![allow(clippy::blanket_clippy_restriction_lints)]
#![allow(clippy::shadow_same)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cargo_common_metadata)]
#![allow(clippy::separated_literal_suffix)]
#![allow(clippy::float_arithmetic)]

use anyhow::{anyhow, Error, Result};
use clap::Parser;
use dirs::home_dir;
use sled::Db;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Whether or not evidence supporting the hypothesis was observed
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Evidence {
    /// Evidence supporting the hypothesis was observed
    Observed,
    /// Evidence supporting the hypothesis was not observed
    NotObserved,
}

impl FromStr for Evidence {
    type Err = Error;

    #[inline]
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "o" | "observed" | "Observed" => Ok(Self::Observed),
            "n" | "not-observed" | "NotObserved" => Ok(Self::NotObserved),
            _ => Err(anyhow!("Invalid evidence: {}", s)),
        }
    }
}

impl Display for Evidence {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Observed => write!(f, "Observed"),
            Self::NotObserved => write!(f, "NotObserved"),
        }
    }
}

/// Whether or not the hypothesis should be updated in the database
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum UpdateHypothesis {
    /// The hypothesis should be updated
    Update,
    /// The hypothesis should not be updated
    NoUpdate,
}

impl FromStr for UpdateHypothesis {
    type Err = Error;

    #[inline]
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "u" | "update" | "Update" => Ok(Self::Update),
            "n" | "no-update" | "NoUpdate" => Ok(Self::NoUpdate),
            _ => Err(anyhow!("Invalid update hypothesis: {}", s)),
        }
    }
}

impl Display for UpdateHypothesis {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Update => write!(f, "Update"),
            Self::NoUpdate => write!(f, "NoUpdate"),
        }
    }
}

/// Arguments for the `ask-bayes` command
#[derive(Parser, Debug)]
#[non_exhaustive]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Name of the Hypothesis to update
    #[clap(short, long)]
    pub name: String,
    /// The prior probability of the hypothesis P(H)
    #[clap(short, long, default_value_t = 0.5, validator = validate_probability)]
    pub prior: f64,
    /// The likelihood of the evidence P(E|H)
    #[clap(short, long, default_value_t = 0.5, validator = validate_probability)]
    pub likelihood: f64,
    /// The likelihood of the evidence P(E|¬H)
    #[clap(long, default_value_t = 0.5, validator = validate_probability)]
    pub likelihood_not: f64,
    /// Indicates whether supporting evidence is observed
    #[clap(short, long, default_value_t = Evidence::Observed)]
    pub evidence: Evidence,
    /// Updates the prior probability of the hypothesis P(H) to the new posterior probability, saving it to the database
    #[clap(short, long, default_value_t = UpdateHypothesis::NoUpdate)]
    pub update_prior: UpdateHypothesis,
    /// Returns the saved value of the prior probability of the hypothesis P(H).
    /// Incompatible with other flags aside from `--name`
    #[clap(
        short,
        long,
        conflicts_with = "prior",
        conflicts_with = "likelihood",
        conflicts_with = "likelihood-not",
        conflicts_with = "evidence",
        conflicts_with = "update-prior"
    )]
    pub get_prior: bool,
    /// Sets the prior probability of the hypothesis P(H) to the new value, saving it to the database.
    /// Incompatible with other flags aside from `--name` and `--prior`
    #[clap(
        short,
        long,
        requires = "prior",
        conflicts_with = "likelihood",
        conflicts_with = "likelihood-not",
        conflicts_with = "evidence",
        conflicts_with = "update-prior",
        conflicts_with = "get-prior"
    )]
    pub set_prior: bool,
    /// Removes the prior probability of the hypothesis P(H) from the database.
    /// Incompatible with other flags aside from `--name`
    #[clap(
        short,
        long,
        conflicts_with = "prior",
        conflicts_with = "likelihood",
        conflicts_with = "likelihood-not",
        conflicts_with = "evidence",
        conflicts_with = "update-prior",
        conflicts_with = "set-prior",
        conflicts_with = "get-prior"
    )]
    pub remove_prior: bool,
}

/// The posterior probability of the hypothesis P(H|E) if the evidence is observed, or P(H|¬E) if the evidence is not observed
#[must_use]
#[inline]
pub fn calculate_posterior_probability(
    prior: f64,
    likelihood: f64,
    likelihood_not: f64,
    evidence: &Evidence,
) -> f64 {
    match *evidence {
        Evidence::Observed => {
            // P(H|E) = P(H) * P(E|H) / (P(H) * P(E|H) + P(¬H) * P(E|¬H))
            likelihood * prior / likelihood.mul_add(prior, likelihood_not * negate(prior))
        }
        Evidence::NotObserved => {
            // P(H|¬E) = P(H) * P(¬E|H) / (P(H) * P(¬E|H) + P(¬H) * P(¬E|¬H))
            negate(likelihood) * prior
                / negate(likelihood).mul_add(prior, negate(likelihood_not) * negate(prior))
        }
    }
}

/// Gets the prior probability of the hypothesis P(H) from the database.
/// # Errors
/// - If the prior probability of the hypothesis is not in the database
/// - If the database cannot be opened
/// - If the prior value is not a valid float  
#[inline]
pub fn get_prior(name: &str) -> Result<f64> {
    let db = open_db()?;
    let prior = db.get(&name)?;
    match prior {
        Some(prior_serialized) => {
            let bytes = prior_serialized.as_ref();
            Ok(f64::from_be_bytes(bytes.try_into()?))
        }
        None => return Err(anyhow!("Could not find hypothesis {name}")),
    }
}

/// Sets the prior probability of the hypothesis P(H) to the new value, saving it to the database.
/// # Errors
/// - If the database cannot be opened
/// - If the prior cannot be inserted into the database
#[inline]
pub fn set_prior(name: &str, prior: f64) -> Result<()> {
    let db = open_db()?;
    db.insert(name, &prior.to_be_bytes())?;
    Ok(())
}

/// Removes the prior probability of the hypothesis P(H) from the database
/// # Errors
/// - If the database cannot be opened
/// - If the prior cannot be removed from the database
#[inline]
pub fn remove_prior(name: &str) -> Result<()> {
    let db = open_db()?;
    db.remove(name)?;
    Ok(())
}

/// Opens the hypotheses database
/// # Errors
/// - If the database cannot be opened
#[inline]
fn open_db() -> Result<Db> {
    let hd = match home_dir() {
        Some(hd) => hd,
        None => return Err(anyhow!("Could not find home directory")),
    };
    let db_path = hd.join(".ask-bayes").join("hypotheses.db");
    Ok(sled::open(db_path)?)
}

/// Validates a probability.  Probabilities should be valid floats between 0 and 1.
fn validate_probability(value: &str) -> Result<f64> {
    let float = value.parse::<f64>()?;
    if !(0.0_f64..=1.0_f64).contains(&float) {
        return Err(anyhow!("Probability must be between 0 and 1"));
    }
    Ok(float)
}

/// Validates the likelihood probabilities.  The sum of the probabilities should be greater than 0.  
/// # Errors
/// - If the sum of the likelihoods is less than or equal to 0 when evidence is observed
/// - If the sum of the negated likelihoods is less than or equal to 0 when evidence is not observed
#[inline]
pub fn validate_likelihoods(
    likelihood: f64,
    likelihood_not: f64,
    evidence: &Evidence,
    name: &str,
) -> Result<()> {
    match *evidence {
        Evidence::Observed => {
            if likelihood + likelihood_not <= 0.0_f64 {
                return Err(anyhow!(
                    "The sum P(E|{name}) + P(E|¬{name}) must be greater than 0 if evidence is observed."
                ));
            }
        }
        Evidence::NotObserved => {
            if negate(likelihood) + negate(likelihood_not) <= 0.0_f64 {
                return Err(anyhow!(
                    "The sum P(¬E|{name}) + P(¬E|¬{name}) must be greater than 0 if evidence is not observed."
                ));
            }
        }
    }

    Ok(())
}

/// Negates a probability.  Ex. P(H) -> P(¬H)
fn negate(value: f64) -> f64 {
    1.0_f64 - value
}
