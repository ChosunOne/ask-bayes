use anyhow::Result;
use ask_bayes::prelude::*;
use clap::Parser;
use log::{info, LevelFilter};
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

#[cfg(not(tarpaulin_include))]
fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Info,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;

    let args = Args::parse();
    if args.wizard {
        wizard()?;
        return Ok(());
    }

    let name = args.name.ok_or(anyhow::anyhow!("name is required"))?;

    if args.get_prior {
        let p = get_prior(&name)?;
        info!("P({name}) = {}", p);
        return Ok(());
    }

    if args.remove_prior {
        remove_prior(&name)?;
        info!("P({name}) removed");
        return Ok(());
    }

    if let Some(prior) = args.set_prior {
        set_prior(&name, prior)?;
        info!("P({name}) = {}", prior);
        return Ok(());
    }

    let prior = args.prior.ok_or(anyhow::anyhow!("prior is required"))?;
    let likelihood = args
        .likelihood
        .ok_or(anyhow::anyhow!("likelihood is required"))?;
    let likelihood_not = args
        .likelihood_null
        .ok_or(anyhow::anyhow!("likelihood_not is required"))?;
    let evidence = args
        .evidence
        .ok_or(anyhow::anyhow!("evidence is required"))?;
    let posterior_probability =
        calculate_posterior_probability(prior, likelihood, likelihood_not, &evidence, &name)?;
    let output_format = args.output.ok_or(anyhow::anyhow!("output is required"))?;

    report_posterior_probability(
        prior,
        likelihood,
        likelihood_not,
        &evidence,
        posterior_probability,
        &name,
        &output_format,
    );

    if let Some(UpdateHypothesis::Update) = args.update_prior {
        set_prior(&name, posterior_probability)?;
        info!("P({name}) has been updated to {}", posterior_probability);
    }
    Ok(())
}
