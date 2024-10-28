# Dual Defense: Enhancing Privacy and Mitigating Poisoning Attacks in Federated Learning

This repository contains the official implementation of the experiments from the NeurIPS 2024 paper: "Dual Defense: Enhancing Privacy and Mitigating Poisoning Attacks in Federated Learning".

## Overview
Federated learning (FL) is inherently susceptible to privacy breaches and poisoning attacks. To tackle these challenges, researchers have separately devised secure aggregation mechanisms to protect data privacy and robust aggregation methods that withstand poisoning attacks. However, simultaneously addressing both concerns is challenging; secure aggregation facilitates poisoning attacks as most anomaly detection techniques require access to unencrypted local model updates, which are obscured by secure aggregation.
Few recent efforts to simultaneously tackle both challenges offen depend on impractical assumption of non-colluding two-server setups that disrupt FL's topology, or three-party computation which introduces scalability issues, complicating deployment and application.
To overcome this dilemma, this paper introduce a \textbf{D}ual \textbf{D}efense \textbf{Fed}erated learning (\textit{DDFed}) framework.
\textit{DDFed} simultaneously boosts privacy protection and mitigates poisoning attacks, without introducing new participant roles or disrupting the existing FL topology.
\textit{DDFed} initially leverages cutting-edge fully homomorphic encryption (FHE) to securely aggregate model updates, without the impractical requirement for non-colluding two-server setups and ensures strong privacy protection. 
Additionally, we proposes a unique two-phase anomaly detection mechanism for encrypted model updates, featuring secure similarity computation and feedback-driven collaborative selection, with additional measures to prevent potential privacy breaches from Byzantine clients incorporated into the detection process.


## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: PyTorch, tenseal, and others specified in requirements.txt.

### Installation
Clone the repository and install the dependencies:

```
git clone git@github.com:irxyzzz/DualDefense.git
cd DualDefense
pip install -r requirements.txt
```
### Usage
To run the experiments, refer to `run_test.sh` for a sample script.

## Citation
If you find this repository useful in your research, please consider citing our work:

@inproceedings{xu2024dualdefense,
  title={Dual Defense: Enhancing Privacy and Mitigating Poisoning Attacks in Federated Learning},
  author={Xu, Runhua and Gao, Shiqi and Li, Chao and Joshi, James and Li, Jianxin},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please open an issue on this repository or contact the authors.
