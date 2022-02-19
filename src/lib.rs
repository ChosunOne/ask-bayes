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
    #[clap(short, long, default_value_t = 0.5, group = "compute")]
    pub prior: f64,
    /// The likelihood of the evidence P(E|H)
    #[clap(short, long, default_value_t = 0.5, group = "compute")]
    pub likelihood: f64,
    /// The likelihood of the evidence P(E|¬H)
    #[clap(long, default_value_t = 0.5, group = "compute")]
    pub likelihood_not: f64,
    /// Indicates supporting evidence is observed
    #[clap(short, long, group = "compute")]
    pub observed_evidence: bool,
    /// Updates the prior probability of the hypothesis P(H) to the new posterior probability, saving it to the database
    #[clap(short, long, group = "compute")]
    pub update_prior: bool,
    /// Returns the saved value of the prior probability of the hypothesis P(H)
    #[clap(short, long, group = "get_prior", conflicts_with = "compute")]
    pub get_prior: bool,
    /// Sets the prior probability of the hypothesis P(H) to the new value, saving it to the database
    #[clap(
        short,
        long,
        group = "set_prior",
        conflicts_with = "compute",
        conflicts_with = "get_prior"
    )]
    pub set_prior: bool,
    /// Removes the prior probability of the hypothesis P(H) from the database
    #[clap(
        short,
        long,
        conflicts_with = "compute",
        conflicts_with = "set_prior",
        conflicts_with = "get_prior"
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
