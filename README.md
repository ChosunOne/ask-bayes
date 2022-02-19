<div align="center">

  <h1>Ask Bayes</h1>

  <p>
    <a href="https://crates.io/crates/ask_bayes">
      <img src="https://img.shields.io/crates/l/ask_bayes?style=for-the-badge" alt="License"/>
    </a>
    <a href="https://docs.rs/{{project-name}}">
      <img src="https://img.shields.io/badge/docs-latest-blue.svg?style=for-the-badge" alt="docs.rs docs" />
    </a>
  </p>

<sub><h4>Built with 🦀</h4></sub>
</div>

# <p id="about">About</p>

Ask Bayes is a simple command line tool for performing Bayesian inference.  You can save your updates locally to preserve the latest priors for your hypotheses.

# <p id="installation">Installation</p>

  <h4>From Cargo</h4>

`cargo install ask-bayes`

# <p id="usage">Usage</p>
Ask Bayes can be invoked like so:  
`ask-bayes -n Hypothesis-name -p 0.75 -l 0.75 -o`  
which will output:  
```bash 
P(Hypothesis-name) = 0.75
P(E|Hypothesis-name) = 0.75
P(E|¬Hypothesis-name) = 0.5
P(Hypothesis-name|E) = 0.8181818181818182
```
see `ask-bayes --help` for more information.

# <p id="license">License</p>

This project is licensed under either of

- Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

# <p id="contribution">Contribution</p>


Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
