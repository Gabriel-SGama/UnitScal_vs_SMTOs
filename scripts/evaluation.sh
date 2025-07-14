# Get plots from the article
python3.8 supervised_experiments/evaluation/show_latex_table.py --dataset mnist --model lenet --tasks CL_RR
python3.8 supervised_experiments/evaluation/plot_results_w_weights.py --tasks CL_CR
python3.8 supervised_experiments/evaluation/plot_results_w_weights.py --tasks RL_RR
python3.8 supervised_experiments/evaluation/plot_grad_metrics.py

# Get table draft and plots
python3.8 supervised_experiments/evaluation/show_latex_table.py --dataset cityscapes --model resnet18 --tasks S_D_I --shape 512 256 --aug
python3.8 supervised_experiments/evaluation/show_latex_table.py --dataset QM9 --model mpnn --tasks mu_alpha_homo_lumo_r2_zpve_U0Atom_UAtom_HAtom_GAtom_Cv

# Fixed weights
python3.8 supervised_experiments/evaluation/show_latex_table.py --dataset QM9 --model mpnn --tasks mu_alpha_homo_lumo_r2_zpve_U0Atom_UAtom_HAtom_GAtom_Cv --analysis_test FwLe0.9
python3.8 supervised_experiments/evaluation/show_nash_weights.py 