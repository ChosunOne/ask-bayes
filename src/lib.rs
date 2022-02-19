use anyhow::{anyhow, Result};
use clap::Parser;
use dirs::home_dir;
use sled::Db;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Name of the Hypothesis to update
    #[clap(short, long)]
    pub name: String,
    /// The prior probability of the hypothesis P(H)
    #[clap(short, long, default_value_t = 0.5)]
    pub prior: f64,
    /// The likelihood of the evidence P(E|H)
    #[clap(short, long, default_value_t = 0.5)]
    pub likelihood: f64,
    /// The likelihood of the evidence P(E|¬H)
    #[clap(long, default_value_t = 0.5)]
    pub likelihood_not: f64,
    /// Indicates supporting evidence is observed
    #[clap(short, long)]
    pub observed_evidence: bool,
    /// Updates the prior probability of the hypothesis P(H) to the new posterior probability, saving it to the database
    #[clap(short, long)]
    pub update_prior: bool,
    /// Returns the saved value of the prior probability of the hypothesis P(H).
    /// Incompatible with other flags aside from `--name`
    #[clap(
        short,
        long,
        conflicts_with = "prior",
        conflicts_with = "likelihood",
        conflicts_with = "likelihood-not",
        conflicts_with = "observed-evidence",
        conflicts_with = "update-prior"
    )]
    pub get_prior: bool,
    /// Sets the prior probability of the hypothesis P(H) to the new value, saving it to the database.
    /// Incompatible with other flags aside from `--name` and `--prior`
    #[clap(
        short,
        long,
        group = "set_prior",
        conflicts_with = "prior",
        conflicts_with = "likelihood",
        conflicts_with = "likelihood-not",
        conflicts_with = "observed-evidence",
        conflicts_with = "update-prior",
        conflicts_with = "get-prior"
    )]
    pub set_prior: bool,
    /// Removes the prior probability of the hypothesis P(H) from the database
    /// Incompatible with other flags aside from `--name`
    #[clap(
        short,
        long,
        conflicts_with = "prior",
        conflicts_with = "likelihood",
        conflicts_with = "likelihood-not",
        conflicts_with = "observed-evidence",
        conflicts_with = "update-prior",
        conflicts_with = "set-prior",
        conflicts_with = "get-prior"
    )]
    pub remove_prior: bool,
}

/// The posterior probability of the hypothesis P(H|E)
pub fn calculate_posterior_probability(
    prior: f64,
    likelihood: f64,
    likelihood_not: f64,
    observed_evidence: bool,
) -> f64 {
    if observed_evidence {
        likelihood * prior / (likelihood * prior + likelihood_not * (1.0 - prior))
    } else {
        (1.0 - likelihood) * prior
            / ((1.0 - likelihood) * prior + (1.0 - likelihood_not) * (1.0 - prior))
    }
}

pub fn get_prior(name: &str) -> Result<f64> {
    let db = open_db()?;
    let prior = db.get(&name)?;
    match prior {
        Some(prior) => {
            let bytes = prior.as_ref();
            Ok(f64::from_be_bytes(bytes.try_into()?))
        }
        None => return Err(anyhow!("Could not find hypothesis {name}")),
    }
}

pub fn set_prior(name: &str, prior: f64) -> Result<()> {
    let db = open_db()?;
    db.insert(name, &prior.to_be_bytes())?;
    Ok(())
}

pub fn remove_prior(name: &str) -> Result<()> {
    let db = open_db()?;
    db.remove(name)?;
    Ok(())
}

fn open_db() -> Result<Db> {
    let hd = match home_dir() {
        Some(hd) => hd,
        None => return Err(anyhow!("Could not find home directory")),
    };
    let db_path = hd.join(".ask-bayes").join("hypotheses.db");
    Ok(sled::open(db_path)?)
}
