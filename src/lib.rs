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

/// The prelude for the `ask-bayes` crate.
pub mod prelude {
    pub use crate::{
        calculate_posterior_probability, get_prior, remove_prior, set_prior, Args, Evidence,
        UpdateHypothesis,
    };
}

/// Whether or not evidence supporting the hypothesis was observed
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
/// # Errors
/// - If the P(E) is 0
#[inline]
pub fn calculate_posterior_probability(
    prior: f64,
    likelihood: f64,
    likelihood_not: f64,
    evidence: &Evidence,
    name: &str,
) -> Result<f64> {
    validate_likelihoods_and_prior(prior, likelihood, likelihood_not, evidence, name)?;
    let p_e = probability_of_observing_evidence(prior, likelihood, likelihood_not);
    match *evidence {
        Evidence::Observed => {
            // P(H|E) = P(H) * P(E|H) / P(E)
            Ok(likelihood * prior / p_e)
        }
        Evidence::NotObserved => {
            // P(H|¬E) = P(H) * P(¬E|H) / P(¬E)
            Ok(negate(likelihood) * prior / negate(p_e))
        }
    }
}

/// Gets the prior probability of the hypothesis P(H) from the database.
/// # Errors
/// - If the prior probability of the hypothesis is not in the database
/// - If the database cannot be opened
/// - If the prior value is not a valid float  
#[inline]
#[cfg(not(tarpaulin_include))]
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
#[cfg(not(tarpaulin_include))]
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
#[cfg(not(tarpaulin_include))]
pub fn remove_prior(name: &str) -> Result<()> {
    let db = open_db()?;
    db.remove(name)?;
    Ok(())
}

/// Opens the hypotheses database
/// # Errors
/// - If the database cannot be opened
#[inline]
#[cfg(not(tarpaulin_include))]
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

/// Negates a probability.  Ex. P(H) -> P(¬H)
fn negate(value: f64) -> f64 {
    1.0_f64 - value
}

/// Checks that P(E) is not 0
fn validate_likelihoods_and_prior(
    prior: f64,
    likelihood: f64,
    likelihood_not: f64,
    evidence: &Evidence,
    name: &str,
) -> Result<()> {
    match *evidence {
        Evidence::Observed => {
            if probability_of_observing_evidence(prior, likelihood, likelihood_not) <= 0.0_f64 {
                return Err(anyhow!("The total probability of observing evidence P(E) must be greater than 0 if evidence is observed.  \r\nP(E) = P({name})[{prior}] * P(E|{name})[{likelihood}] + P(\u{ac}{name})[{}] * P(E|\u{ac}{name})[{}] = 0", negate(prior), likelihood_not));
            }
        }
        Evidence::NotObserved => {
            if negate(probability_of_observing_evidence(
                prior,
                likelihood,
                likelihood_not,
            )) <= 0.0_f64
            {
                return Err(anyhow!("The total probability of not observing evidence P(\u{ac}E) must be greater than 0 if evidence is not observed.  \r\nP(\u{ac}E) = P(\u{ac}E|{name})[{}] * P({name})[{prior}] + P(\u{ac}{name})[{}] * P(\u{ac}E|\u{ac}{name})[{}] = 0", negate(likelihood), negate(prior), negate(likelihood_not)));
            }
        }
    }

    Ok(())
}

///  P(H) * P(E|H) + P(¬H) * P(E|¬H)
fn probability_of_observing_evidence(prior: f64, likelihood: f64, likelihood_not: f64) -> f64 {
    likelihood.mul_add(prior, likelihood_not * negate(prior))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_validates_a_valid_probability() -> Result<()> {
        let prob = "0.75";
        let result = validate_probability(prob)?;
        assert_eq!(result, 0.75_f64);
        Ok(())
    }

    #[test]
    fn it_fails_to_validate_a_probability_greater_than_1() -> Result<()> {
        let prob = "1.1";
        let result = validate_probability(prob);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn it_fails_to_validate_a_probability_less_than_0() -> Result<()> {
        let prob = "-0.1";
        let result = validate_probability(prob);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn it_fails_to_validate_an_invalid_float() -> Result<()> {
        let prob = "invalid";
        let result = validate_probability(prob);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn it_validates_a_valid_pair_of_likelihoods() -> Result<()> {
        let likelihood = 0.75;
        let likelihood_not = 0.25;
        let prior = 0.5;
        let evidence = Evidence::Observed;
        let name = "test";
        validate_likelihoods_and_prior(prior, likelihood, likelihood_not, &evidence, name)?;
        Ok(())
    }

    #[test]
    fn it_validates_a_valid_pair_of_negated_likelihoods() -> Result<()> {
        let prior = 0.5;
        let likelihood = 0.75;
        let likelihood_not = 0.25;
        let evidence = Evidence::NotObserved;
        let name = "test";
        validate_likelihoods_and_prior(prior, likelihood, likelihood_not, &evidence, name)?;
        Ok(())
    }

    #[test]
    fn it_fails_to_validate_a_pair_of_likelihoods_with_evidence_observed_when_the_sum_is_less_than_or_equal_to_0(
    ) -> Result<()> {
        let prior = 0.5;
        let likelihood = 0.0;
        let likelihood_not = 0.0;
        let evidence = Evidence::Observed;
        let name = "test";
        let result =
            validate_likelihoods_and_prior(prior, likelihood, likelihood_not, &evidence, name);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn it_fails_to_validate_a_pair_of_negated_likelihoods_with_evidence_not_observed_when_the_negated_sum_is_less_than_or_equal_to_0(
    ) -> Result<()> {
        let prior = 0.5;
        let likelihood = 1.0;
        let likelihood_not = 1.0;
        let evidence = Evidence::NotObserved;
        let name = "test";
        let result =
            validate_likelihoods_and_prior(prior, likelihood, likelihood_not, &evidence, name);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn it_parses_a_valid_evidence_string() -> Result<()> {
        {
            let evidence = "observed";
            let result = Evidence::from_str(evidence)?;
            assert_eq!(result, Evidence::Observed);
        }
        {
            let evidence = "o";
            let result = Evidence::from_str(evidence)?;
            assert_eq!(result, Evidence::Observed);
        }
        {
            let evidence = "Observed";
            let result = Evidence::from_str(evidence)?;
            assert_eq!(result, Evidence::Observed);
        }
        {
            let evidence = "not-observed";
            let result = Evidence::from_str(evidence)?;
            assert_eq!(result, Evidence::NotObserved);
        }
        {
            let evidence = "n";
            let result = Evidence::from_str(evidence)?;
            assert_eq!(result, Evidence::NotObserved);
        }
        {
            let evidence = "NotObserved";
            let result = Evidence::from_str(evidence)?;
            assert_eq!(result, Evidence::NotObserved);
        }
        Ok(())
    }

    #[test]
    fn it_fails_to_parse_an_invalid_evidence_string() -> Result<()> {
        let evidence = "invalid";
        let result = Evidence::from_str(evidence);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn it_displays_a_valid_evidence_string() -> Result<()> {
        {
            let evidence = Evidence::Observed;
            let result = evidence.to_string();
            assert_eq!(result, "Observed");
        }
        {
            let evidence = Evidence::NotObserved;
            let result = evidence.to_string();
            assert_eq!(result, "NotObserved");
        }
        Ok(())
    }

    #[test]
    fn it_parses_a_valid_update_string() -> Result<()> {
        {
            let update = "update";
            let result = UpdateHypothesis::from_str(update)?;
            assert_eq!(result, UpdateHypothesis::Update);
        }
        {
            let update = "u";
            let result = UpdateHypothesis::from_str(update)?;
            assert_eq!(result, UpdateHypothesis::Update);
        }
        {
            let update = "Update";
            let result = UpdateHypothesis::from_str(update)?;
            assert_eq!(result, UpdateHypothesis::Update);
        }
        {
            let update = "no-update";
            let result = UpdateHypothesis::from_str(update)?;
            assert_eq!(result, UpdateHypothesis::NoUpdate);
        }
        {
            let update = "n";
            let result = UpdateHypothesis::from_str(update)?;
            assert_eq!(result, UpdateHypothesis::NoUpdate);
        }
        {
            let update = "NoUpdate";
            let result = UpdateHypothesis::from_str(update)?;
            assert_eq!(result, UpdateHypothesis::NoUpdate);
        }
        Ok(())
    }

    #[test]
    fn it_fails_to_parse_an_invalid_update_string() -> Result<()> {
        let update = "invalid";
        let result = UpdateHypothesis::from_str(update);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn it_displays_a_valid_update_string() -> Result<()> {
        {
            let update = UpdateHypothesis::Update;
            let result = update.to_string();
            assert_eq!(result, "Update");
        }
        {
            let update = UpdateHypothesis::NoUpdate;
            let result = update.to_string();
            assert_eq!(result, "NoUpdate");
        }
        Ok(())
    }

    #[test]
    fn it_calculates_the_posterior_probability_when_evidence_is_observed() -> Result<()> {
        let prior = 0.75;
        let likelihood = 0.75;
        let likelihood_not = 0.5;
        let evidence = Evidence::Observed;
        let name = "test";
        let result =
            calculate_posterior_probability(prior, likelihood, likelihood_not, &evidence, name)?;
        assert_eq!(result, 0.8181818181818182);
        Ok(())
    }

    #[test]
    fn it_calculates_the_posterior_probability_when_evidence_is_not_observed() -> Result<()> {
        let prior = 0.75;
        let likelihood = 0.75;
        let likelihood_not = 0.5;
        let evidence = Evidence::NotObserved;
        let name = "test";
        let result =
            calculate_posterior_probability(prior, likelihood, likelihood_not, &evidence, name)?;
        assert_eq!(result, 0.6);
        Ok(())
    }

    #[test]
    fn it_fails_to_validate_likelihoods_and_hypothesis_when_the_negated_prior_is_zero() -> Result<()>
    {
        let name = "test";
        let prior = 1.0;
        let likelihood = 1.0;
        let likelihood_not = 0.5;
        let evidence = Evidence::NotObserved;
        let result =
            validate_likelihoods_and_prior(prior, likelihood, likelihood_not, &evidence, name);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn it_fails_to_validate_likelihoods_and_hypothesis_when_the_prior_is_zero() -> Result<()> {
        let name = "test";
        let prior = 0.0;
        let likelihood = 0.5;
        let likelihood_not = 0.0;
        let evidence = Evidence::Observed;
        let result =
            validate_likelihoods_and_prior(prior, likelihood, likelihood_not, &evidence, name);
        assert!(result.is_err());
        Ok(())
    }
}
