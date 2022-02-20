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
#![allow(clippy::struct_excessive_bools)]

use anyhow::{anyhow, Error, Result};
use clap::Parser;
use dialoguer::Input;
use dirs::home_dir;
use log::info;
use prettytable::{format, Cell, Row, Table};
use serde_json::json;
use sled::Db;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// The prelude for the `ask-bayes` crate.
pub mod prelude {
    pub use crate::{
        calculate_posterior_probability, get_prior, remove_prior, report_posterior_probability,
        set_prior, wizard, Args, Evidence, UpdateHypothesis,
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
            "o" | "observed" | "Observed" | "y" | "Y" => Ok(Self::Observed),
            "n" | "not-observed" | "NotObserved" | "N" | "not observed" => Ok(Self::NotObserved),
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
            "u" | "update" | "Update" | "y" | "Y" => Ok(Self::Update),
            "n" | "no-update" | "NoUpdate" | "N" => Ok(Self::NoUpdate),
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

/// The output format to use when displaying results to the user
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputFormat {
    /// Output in a table format
    Table,
    /// Output in a JSON format
    Json,
    /// Output in a formatted string
    Simple,
}

impl FromStr for OutputFormat {
    type Err = Error;

    #[inline]
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "table" | "Table" | "t" | "T" => Ok(Self::Table),
            "json" | "Json" | "j" | "J" => Ok(Self::Json),
            "simple" | "Simple" | "s" | "S" => Ok(Self::Simple),
            _ => Err(anyhow!("Invalid output format: {}", s)),
        }
    }
}

impl Display for OutputFormat {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Table => write!(f, "Table"),
            Self::Json => write!(f, "Json"),
            Self::Simple => write!(f, "Simple"),
        }
    }
}

/// Arguments for the `ask-bayes` command
#[derive(Parser, Debug)]
#[non_exhaustive]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Name of the Hypothesis to update
    #[clap(
        short,
        long,
        forbid_empty_values = true,
        required_unless_present("wizard")
    )]
    pub name: Option<String>,
    /// The prior probability of the hypothesis P(H)
    #[clap(
        short,
        long,
        default_value_if("name", None, Some("0.5")),
        validator = parse_validate_probability,
        forbid_empty_values = true,
        required_unless_present("wizard")
    )]
    pub prior: Option<f64>,
    /// The likelihood of the evidence P(E|H)
    #[clap(
        short,
        long,
        default_value_if("name", None, Some("0.5")),
        validator = parse_validate_probability,
        forbid_empty_values = true,
        required_unless_present("wizard"))]
    pub likelihood: Option<f64>,
    /// The likelihood of the evidence P(E|¬H)
    #[clap(
        long,
        default_value_if("name", None, Some("0.5")),
        validator = parse_validate_probability,
        forbid_empty_values = true,
        required_unless_present("wizard"))]
    pub likelihood_null: Option<f64>,
    /// Indicates whether supporting evidence is observed
    #[clap(
        short,
        long,
        default_value_if("name", None, Some("Observed")),
        default_missing_value = "Observed",
        possible_values = ["o", "observed", "Observed", "n", "not-observed", "NotObserved"],
        required_unless_present("wizard"))]
    pub evidence: Option<Evidence>,
    /// Updates the prior probability of the hypothesis P(H) to the new posterior probability, saving it to the database
    #[clap(
        short,
        long,
        default_value_if("name", None, Some("NoUpdate")),
        default_missing_value = "Update",
        possible_values = ["u", "update", "Update", "n", "no-update", "NoUpdate"])]
    pub update_prior: Option<UpdateHypothesis>,
    /// Returns the saved value of the prior probability of the hypothesis P(H).
    /// Incompatible with other flags aside from `--name`
    #[clap(
        short,
        long,
        conflicts_with = "prior",
        conflicts_with = "likelihood",
        conflicts_with = "likelihood-null",
        conflicts_with = "evidence",
        conflicts_with = "update-prior"
    )]
    pub get_prior: bool,
    /// Sets the prior probability of the hypothesis P(H) to the new value, saving it to the database.
    /// Incompatible with other flags aside from `--name` and `--prior`
    #[clap(
        short,
        long,
        default_value_if("name", None, Some("0.5")),
        validator = parse_validate_probability,
        conflicts_with = "prior",
        conflicts_with = "likelihood",
        conflicts_with = "likelihood-null",
        conflicts_with = "evidence",
        conflicts_with = "update-prior",
        conflicts_with = "get-prior"
    )]
    pub set_prior: Option<f64>,
    /// Removes the prior probability of the hypothesis P(H) from the database.
    /// Incompatible with other flags aside from `--name`
    #[clap(
        short,
        long,
        conflicts_with = "prior",
        conflicts_with = "likelihood",
        conflicts_with = "likelihood-null",
        conflicts_with = "evidence",
        conflicts_with = "update-prior",
        conflicts_with = "set-prior",
        conflicts_with = "get-prior"
    )]
    pub remove_prior: bool,
    /// Runs the wizard to help guide you through the process of updating a hypothesis
    #[clap(short, long, exclusive = true, takes_value = false)]
    pub wizard: bool,
    /// The type of output to display
    #[clap(
        short,
        long,
        default_value_if("name", None, Some("Table")),
        possible_values = ["t", "table", "Table", "T", "j", "json", "Json", "J", "s", "simple", "Simple", "S"],
        required_unless_present("wizard")
    )]
    pub output: Option<OutputFormat>,
}

/// The posterior probability of the hypothesis P(H|E) if the evidence is observed, or P(H|¬E) if the evidence is not observed
/// # Errors
/// - If the P(E) is 0
#[inline]
pub fn calculate_posterior_probability(
    prior: f64,
    likelihood: f64,
    likelihood_null: f64,
    evidence: &Evidence,
    name: &str,
) -> Result<f64> {
    validate_likelihoods_and_prior(prior, likelihood, likelihood_null, evidence, name)?;
    let p_e = marginal_likelihood(prior, likelihood, likelihood_null);
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

/// Parses and validates a probability
fn parse_validate_probability(value: &str) -> Result<f64> {
    let float = value.parse::<f64>()?;
    validate_probability(float)?;
    Ok(float)
}

/// Validates a probability.  Probabilities should be valid floats between 0 and 1.
fn validate_probability(value: f64) -> Result<()> {
    if !(0.0_f64..=1.0_f64).contains(&value) {
        return Err(anyhow!("Probability must be between 0 and 1"));
    }
    Ok(())
}

/// Negates a probability.  Ex. P(H) -> P(¬H)
fn negate(value: f64) -> f64 {
    1.0_f64 - value
}

/// Checks that P(E) is not 0
fn validate_likelihoods_and_prior(
    prior: f64,
    likelihood: f64,
    likelihood_null: f64,
    evidence: &Evidence,
    name: &str,
) -> Result<()> {
    match *evidence {
        Evidence::Observed => {
            if marginal_likelihood(prior, likelihood, likelihood_null) <= 0.0_f64 {
                return Err(anyhow!("The total probability of observing evidence P(E) must be greater than 0 if evidence is observed.  \r\nP(E) = P({name})[{prior}] * P(E|{name})[{likelihood}] + P(\u{ac}{name})[{}] * P(E|\u{ac}{name})[{}] = 0", negate(prior), likelihood_null));
            }
        }
        Evidence::NotObserved => {
            if negate(marginal_likelihood(prior, likelihood, likelihood_null)) <= 0.0_f64 {
                return Err(anyhow!("The total probability of not observing evidence P(\u{ac}E) must be greater than 0 if evidence is not observed.  \r\nP(\u{ac}E) = P(\u{ac}E|{name})[{}] * P({name})[{prior}] + P(\u{ac}{name})[{}] * P(\u{ac}E|\u{ac}{name})[{}] = 0", negate(likelihood), negate(prior), negate(likelihood_null)));
            }
        }
    }

    Ok(())
}

///  P(H) * P(E|H) + P(¬H) * P(E|¬H), otherwise known as P(E)
fn marginal_likelihood(prior: f64, likelihood: f64, likelihood_null: f64) -> f64 {
    likelihood.mul_add(prior, likelihood_null * negate(prior))
}

/// Runs the wizard to guide the update of the prior probability of the hypothesis
/// # Errors
/// - If the prompt cannot be displayed
#[inline]
#[cfg(not(tarpaulin_include))]
pub fn wizard() -> Result<()> {
    let name = Input::<String>::new()
        .with_prompt("Enter the name of the hypothesis")
        .allow_empty(false)
        .interact_text()?;

    let prior = Input::<f64>::new()
        .with_prompt(format!(
            "Enter the prior probability of the hypothesis P({name})"
        ))
        .allow_empty(false)
        .default(0.5_f64)
        .validate_with(|v: &f64| validate_probability(*v))
        .interact_text()?;

    let likelihood = Input::<f64>::new()
        .with_prompt(format!(
            "Enter the likelihood of observing evidence given {name} is true P(E|{name})"
        ))
        .allow_empty(false)
        .default(0.5_f64)
        .validate_with(|v: &f64| validate_probability(*v))
        .interact_text()?;

    let likelihood_null = Input::<f64>::new()
        .with_prompt(format!(
            "Enter the likelihood of observing evidence given {name} is false P(E|\u{ac}{name})"
        ))
        .allow_empty(false)
        .default(0.5_f64)
        .validate_with(|v: &f64| validate_probability(*v))
        .interact_text()?;

    let evidence = Input::<Evidence>::new()
        .with_prompt("Is evidence observed or not observed?".to_owned())
        .allow_empty(false)
        .default(Evidence::Observed)
        .interact_text()?;

    let posterior_probability =
        calculate_posterior_probability(prior, likelihood, likelihood_null, &evidence, &name)?;

    let output_format = Input::<OutputFormat>::new()
        .with_prompt("How would you like the output?".to_owned())
        .allow_empty(false)
        .default(OutputFormat::Table)
        .interact_text()?;

    report_posterior_probability(
        prior,
        likelihood,
        likelihood_null,
        &evidence,
        posterior_probability,
        &name,
        &output_format,
    );

    let update = Input::<UpdateHypothesis>::new()
        .with_prompt("Would you like to update the prior probability?".to_owned())
        .allow_empty(false)
        .default(UpdateHypothesis::NoUpdate)
        .interact_text()?;

    if update == UpdateHypothesis::Update {
        set_prior(&name, posterior_probability)?;
        info!("P({name}) has been updated to {}", posterior_probability);
    }

    Ok(())
}

/// Reports the posterior probability of the hypothesis given the evidence.  Also reports the values of the `prior`, `likelihood`, and `likelihood_null`.
#[inline]
#[cfg(not(tarpaulin_include))]
pub fn report_posterior_probability(
    prior: f64,
    likelihood: f64,
    likelihood_null: f64,
    evidence: &Evidence,
    posterior_probability: f64,
    name: &str,
    output_format: &OutputFormat,
) {
    match *output_format {
        OutputFormat::Table => {
            report_table(
                name,
                prior,
                likelihood,
                likelihood_null,
                evidence,
                posterior_probability,
            );
        }
        OutputFormat::Json => {
            report_json(
                name,
                prior,
                likelihood,
                likelihood_null,
                evidence,
                posterior_probability,
            );
        }
        OutputFormat::Simple => {
            let output = format!(
                "
                P({name}) = {prior}
                P(E|{name}) = {likelihood}
                P(E|\u{ac}{name}) = {likelihood_null}
                P({name}|E) = {posterior_probability}
                "
            );
            info!("{output}");
        }
    }
}

/// Reports the posterior probability of the hypothesis given the evidence in a table format.
#[cfg(not(tarpaulin_include))]
fn report_table(
    name: &str,
    prior: f64,
    likelihood: f64,
    likelihood_null: f64,
    evidence: &Evidence,
    posterior_probability: f64,
) {
    let marginal_likelihood = marginal_likelihood(prior, likelihood, likelihood_null);
    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
    table.set_titles(Row::new(vec![
        Cell::new("Name"),
        Cell::new("Probability"),
        Cell::new("Value"),
    ]));
    table.add_row(Row::new(vec![
        Cell::new("Prior"),
        Cell::new(&format!("P({name})")),
        Cell::new(&format!("{prior}")),
    ]));
    table.add_row(Row::new(vec![
        Cell::new("Likelihood"),
        Cell::new(&format!("P(E|{name})")),
        Cell::new(&format!("{likelihood}")),
    ]));
    table.add_row(Row::new(vec![
        Cell::new("Likelihood Null"),
        Cell::new(&format!("P(E|\u{ac}{name})")),
        Cell::new(&format!("{likelihood_null}")),
    ]));
    table.add_row(Row::new(vec![
        Cell::new("Marginal Likelihood"),
        Cell::new("P(E)"),
        Cell::new(&format!("{marginal_likelihood}")),
    ]));

    match *evidence {
        Evidence::Observed => table.add_row(Row::new(vec![
            Cell::new("Posterior Probability"),
            Cell::new(&format!("P({name}|E)")),
            Cell::new(&format!("{posterior_probability}")),
        ])),
        Evidence::NotObserved => table.add_row(Row::new(vec![
            Cell::new(&format!("P({name}|\u{ac}E)")),
            Cell::new(&format!("{posterior_probability}")),
        ])),
    };

    table.printstd();
}

/// Reports the posterior probability of the hypothesis given the evidence in a JSON format.
#[cfg(not(tarpaulin_include))]
fn report_json(
    name: &str,
    prior: f64,
    likelihood: f64,
    likelihood_null: f64,
    evidence: &Evidence,
    posterior_probability: f64,
) {
    let json = json!({
        "name": name,
        "prior": prior,
        "likelihood": likelihood,
        "likelihood_null": likelihood_null,
        "evidence": match *evidence {
            Evidence::Observed => "observed",
            Evidence::NotObserved => "not observed",
        },
        "posterior_probability": posterior_probability,
    });

    info!("{}", json.to_string());
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod tests {
    use super::*;

    fn epsilon_compare(a: f64, b: f64) -> bool {
        (a - b).abs() < f64::EPSILON
    }

    #[test]
    fn it_validates_a_valid_probability() -> Result<()> {
        let prob = "0.75";
        let result = parse_validate_probability(prob)?;
        assert!(epsilon_compare(result, 0.75_f64));
        Ok(())
    }

    #[test]
    fn it_fails_to_validate_a_probability_greater_than_1() {
        let prob = "1.1";
        let result = parse_validate_probability(prob);
        assert!(result.is_err());
    }

    #[test]
    fn it_fails_to_validate_a_probability_less_than_0() {
        let prob = "-0.1";
        let result = parse_validate_probability(prob);
        assert!(result.is_err());
    }

    #[test]
    fn it_fails_to_validate_an_invalid_float() {
        let prob = "invalid";
        let result = parse_validate_probability(prob);
        assert!(result.is_err());
    }

    #[test]
    fn it_validates_a_valid_pair_of_likelihoods() -> Result<()> {
        let likelihood = 0.75_f64;
        let likelihood_null = 0.25_f64;
        let prior = 0.5_f64;
        let evidence = Evidence::Observed;
        let name = "test";
        validate_likelihoods_and_prior(prior, likelihood, likelihood_null, &evidence, name)
    }

    #[test]
    fn it_validates_a_valid_pair_of_negated_likelihoods() -> Result<()> {
        let prior = 0.5_f64;
        let likelihood = 0.75_f64;
        let likelihood_null = 0.25_f64;
        let evidence = Evidence::NotObserved;
        let name = "test";
        validate_likelihoods_and_prior(prior, likelihood, likelihood_null, &evidence, name)
    }

    #[test]
    fn it_fails_to_validate_a_pair_of_likelihoods_with_evidence_observed_when_the_sum_is_less_than_or_equal_to_0(
    ) {
        let prior = 0.5_f64;
        let likelihood = 0.0_f64;
        let likelihood_null = 0.0_f64;
        let evidence = Evidence::Observed;
        let name = "test";
        let result =
            validate_likelihoods_and_prior(prior, likelihood, likelihood_null, &evidence, name);
        assert!(result.is_err());
    }

    #[test]
    fn it_fails_to_validate_a_pair_of_negated_likelihoods_with_evidence_not_observed_when_the_negated_sum_is_less_than_or_equal_to_0(
    ) {
        let prior = 0.5_f64;
        let likelihood = 1.0_f64;
        let likelihood_null = 1.0_f64;
        let evidence = Evidence::NotObserved;
        let name = "test";
        let result =
            validate_likelihoods_and_prior(prior, likelihood, likelihood_null, &evidence, name);
        assert!(result.is_err());
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
    fn it_fails_to_parse_an_invalid_evidence_string() {
        let evidence = "invalid";
        let result = Evidence::from_str(evidence);
        assert!(result.is_err());
    }

    #[test]
    fn it_displays_a_valid_evidence_string() {
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
    fn it_fails_to_parse_an_invalid_update_string() {
        let update = "invalid";
        let result = UpdateHypothesis::from_str(update);
        assert!(result.is_err());
    }

    #[test]
    fn it_displays_a_valid_update_string() {
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
    }

    #[test]
    fn it_calculates_the_posterior_probability_when_evidence_is_observed() -> Result<()> {
        let prior = 0.75_f64;
        let likelihood = 0.75_f64;
        let likelihood_null = 0.5_f64;
        let evidence = Evidence::Observed;
        let name = "test";
        let result =
            calculate_posterior_probability(prior, likelihood, likelihood_null, &evidence, name)?;
        assert!(epsilon_compare(result, 0.818_181_818_181_818_2_f64));
        Ok(())
    }

    #[test]
    fn it_calculates_the_posterior_probability_when_evidence_is_not_observed() -> Result<()> {
        let prior = 0.75_f64;
        let likelihood = 0.75_f64;
        let likelihood_null = 0.5_f64;
        let evidence = Evidence::NotObserved;
        let name = "test";
        let result =
            calculate_posterior_probability(prior, likelihood, likelihood_null, &evidence, name)?;
        assert!(epsilon_compare(result, 0.6));
        Ok(())
    }

    #[test]
    fn it_fails_to_validate_likelihoods_and_hypothesis_when_the_negated_prior_is_zero() {
        let name = "test";
        let prior = 1.0_f64;
        let likelihood = 1.0_f64;
        let likelihood_null = 0.5_f64;
        let evidence = Evidence::NotObserved;
        let result =
            validate_likelihoods_and_prior(prior, likelihood, likelihood_null, &evidence, name);
        assert!(result.is_err());
    }

    #[test]
    fn it_fails_to_validate_likelihoods_and_hypothesis_when_the_prior_is_zero() {
        let name = "test";
        let prior = 0.0_f64;
        let likelihood = 0.5_f64;
        let likelihood_null = 0.0_f64;
        let evidence = Evidence::Observed;
        let result =
            validate_likelihoods_and_prior(prior, likelihood, likelihood_null, &evidence, name);
        assert!(result.is_err());
    }

    #[test]
    fn it_parses_a_valid_output_format() -> Result<()> {
        {
            let format = "json";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Json);
        }
        {
            let format = "j";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Json);
        }
        {
            let format = "Json";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Json);
        }
        {
            let format = "J";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Json);
        }
        {
            let format = "t";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Table);
        }
        {
            let format = "Table";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Table);
        }
        {
            let format = "table";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Table);
        }
        {
            let format = "T";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Table);
        }
        {
            let format = "simple";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Simple);
        }
        {
            let format = "s";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Simple);
        }
        {
            let format = "S";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Simple);
        }
        {
            let format = "simple";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Simple);
        }
        {
            let format = "Simple";
            let result = OutputFormat::from_str(format)?;
            assert_eq!(result, OutputFormat::Simple);
        }

        Ok(())
    }

    #[test]
    fn it_fails_to_parse_an_invalid_output_format() {
        let format = "invalid";
        let result = OutputFormat::from_str(format);
        assert!(result.is_err());
    }

    #[test]
    fn it_displays_a_valid_output_format() {
        {
            let format = OutputFormat::Json;
            let result = format.to_string();
            assert_eq!(result, "Json");
        }
        {
            let format = OutputFormat::Table;
            let result = format.to_string();
            assert_eq!(result, "Table");
        }
        {
            let format = OutputFormat::Simple;
            let result = format.to_string();
            assert_eq!(result, "Simple");
        }
    }
}
