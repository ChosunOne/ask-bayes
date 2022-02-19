use ask_bayes::{calculate_posterior_probability, get_prior, remove_prior, set_prior, Args};
use clap::Parser;

fn main() {
    let args = Args::parse();
    if args.get_prior {
        match get_prior(&args.name) {
            Ok(p) => println!("P({}) = {}", args.name, p),
            Err(e) => println!("{}", e),
        }
        return;
    }
    if args.set_prior {
        match set_prior(&args.name, args.prior) {
            Ok(()) => println!("P({}) = {}", args.name, args.prior),
            Err(e) => println!("{}", e),
        }
        return;
    }
    if args.remove_prior {
        match remove_prior(&args.name) {
            Ok(()) => println!("P({}) removed", args.name),
            Err(e) => println!("{}", e),
        }
        return;
    }
    let posterior_probability = calculate_posterior_probability(
        args.prior,
        args.likelihood,
        args.likelihood_not,
        args.observed_evidence,
    );

    println!("P({}) = {}", args.name, args.prior);
    println!("P(E|{}) = {}", args.name, args.likelihood);
    println!("P(E|¬{}) = {}", args.name, args.likelihood_not);
    println!("P({}|E) = {}", args.name, posterior_probability);

    if args.update_prior {
        match set_prior(&args.name, posterior_probability) {
            Ok(()) => println!(
                "P({}) has been updated to {}",
                args.name, posterior_probability
            ),
            Err(e) => println!("{}", e),
        }
    }
}
