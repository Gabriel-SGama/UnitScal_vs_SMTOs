# Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning

Specialized Multi-Task Optimizers (SMTOs) balance task learning in Multi-Task Learning by addressing issues like conflicting gradients and differing gradient norms, which hinder equal-weighted task training. However, recent critiques suggest that equally weighted tasks can achieve competitive results compared to SMTOs, arguing that previous SMTO results were influenced by poor hyperparameter optimization and lack of regularization. In this work, we evaluate these claims through an extensive empirical evaluation of SMTOs, including some of the latest methods, on more complex multi-task problems to clarify this behavior. Our findings indicate that SMTOs perform well compared to uniform loss and that fixed weights can achieve competitive performance compared to SMTOs. Furthermore, we demonstrate why uniform loss perform similarly to SMTOs in some instances.

## Setup

Install [docker](https://docs.docker.com/) and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).

Build docker:

```
./supervised_experiments/build_supervised_docker.sh
```

Update `supervised_docker_run.sh` file with the correct location to the [Cityscapes](https://www.cityscapes-dataset.com/) dataset (with disparity labels). [MNIST](https://yann.lecun.com/exdb/mnist/) and [QM9](http://quantum-machine.org/datasets/) datasets are also used, but they are installed automatically. 

Docker run:

```
./supervised_docker_run.sh 
```

## Preparing the data
The splits used are in txt format on the dataset folder. If `supervised_experiments/loaders/city_preprocess.py` fails, it is necessary to delete the `disparity`, `image`, `instance` or `semantic` folder from `datasets/city_preprocess*` to correctly create the preprocessed dataset.
```
python3.8 supervised_experiments/loaders/multi_mnist_loader.py
python3.8 supervised_experiments/loaders/city_preprocess.py
python3.8 supervised_experiments/loaders/QM9.py
```

## Replicating the article's visualization results

Download all previously trained [data]() and save it on the same folder of the repository.

Run the evaluation scripts for the single-task reference and the SMTOs
```
python3.8 supervised_experiments/evaluation/eval_single_task.py 
python3.8 supervised_experiments/evaluation/eval_multi_task.py
python3.8 supervised_experiments/evaluation/eval_multi_task.py --analysis_test FwLe0.9
python3.8 supervised_experiments/evaluation/eval_multi_task.py --analysis_test grad_metrics
```

Run visualization scripts:

### MNIST
```
python3.8 supervised_experiments/evaluation/show_latex_table.py --dataset mnist --model lenet --tasks CL_RR --rm_optimizers fixed_weights_cagrad fixed_weights_edm --save
python3.8 supervised_experiments/evaluation/plot_results_w_all_weights.py --tasks CL_CR
python3.8 supervised_experiments/evaluation/plot_results_w_all_weights.py --tasks RL_RR
python3.8 supervised_experiments/evaluation/plot_weights.py --dataset mnist
python3.8 supervised_experiments/evaluation/show_latex_table.py --dataset mnist --model lenet --tasks CL_RR --eval_optimizers fixed_weights_cagrad
```

### Cityscapes & Complexity Analysis
```
python3.8 supervised_experiments/evaluation/show_latex_table.py --dataset cityscapes --model resnet18 --tasks S_D_I --shape 512 256 --aug --eval_optimizers baseline auto_lambda cagrad cdtt edm famo imtl nash rotograd --save
python3.8 supervised_experiments/evaluation/plot_grad_metrics.py
python3.8 supervised_experiments/evaluation/plot_grad_metrics.py --all
python3.8 supervised_experiments/evaluation/plot_weights.py --dataset cityscapes
python3.8 supervised_experiments/evaluation/show_latex_table.py --dataset cityscapes --model resnet18 --tasks S_D_I --shape 512 256 --aug --eval_optimizers fixed_weights_edm
```

### QM9
```
python3.8 supervised_experiments/evaluation/show_latex_table.py --dataset QM9 --model mpnn --tasks mu_alpha_homo_lumo_r2_zpve_U0Atom_UAtom_HAtom_GAtom_Cv --eval_optimizers baseline auto_lambda cagrad cdtt edm famo nash --save
python3.8 supervised_experiments/evaluation/show_latex_table.py --dataset QM9 --model mpnn --tasks mu_alpha_homo_lumo_r2_zpve_U0Atom_UAtom_HAtom_GAtom_Cv --analysis_test FwLe0.9 --column half --save
python3.8 supervised_experiments/evaluation/show_nash_weights.py

```

## SMTOs implemented
[**Unit-Scal**](https://arxiv.org/abs/2201.04122): In Defense of the Unitary Scalarization for Deep Multi-Task Learning [[code](https://github.com/yobibyte/unitary-scalarization-dmtl)]

[**Auto-Lambda**](https://arxiv.org/abs/2202.03091): Auto-Lambda: Disentangling Dynamic Task Relationships [[code](https://github.com/lorenmt/auto-lambda)]

[**CAGrad**](https://arxiv.org/abs/2110.14048): Conflict-Averse Gradient Descent for Multi-task Learning [[code](https://github.com/Cranial-XIX/CAGrad)]

[**CDTT**](https://arxiv.org/abs/2204.06698): Leveraging convergence behavior to balance conflicting tasks in multi-task learning [[code](https://github.com/tiemink/MTL_TaskTensioner)]

[**EDM**](https://arxiv.org/abs/2007.06937): Follow the bisector: a simple method for multi-objective optimization [[code](https://github.com/amkatrutsa/edm)]

[**FAMO**](https://arxiv.org/abs/2306.03792): FAMO: Fast Adaptive Multitask Optimization [[code](https://github.com/Cranial-XIX/FAMO)]

[**GradDrop**](https://arxiv.org/abs/2010.06808): Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout [[code](https://github.com/tensorflow/lingvo/blob/master/lingvo/core/graddrop.py)]

[**IMTL**](https://openreview.net/forum?id=IMPnRXEWpvr): Towards Impartial Multi-task Learning

[**MGDA**](https://arxiv.org/abs/1810.04650): Multi-Task Learning as Multi-Objective Optimization [[code](https://github.com/isl-org/MultiObjectiveOptimization)]

[**Nash-MTL** & **SI**](https://arxiv.org/abs/2202.01017): Multi-Task Learning as a Bargaining Game [[code](https://github.com/AvivNavon/nash-mtl)]

[**PCGrad**](https://arxiv.org/abs/2001.06782): Gradient Surgery for Multi-Task Learning [[code](https://github.com/tianheyu927/PCGrad)]

[**RLW**](https://arxiv.org/abs/2111.10603): Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning

[**RotoGrad**](https://arxiv.org/abs/2103.02631): RotoGrad: Gradient Homogenization in Multitask Learning [[code](https://github.com/adrianjav/rotograd)]

[**UW**](https://arxiv.org/abs/1705.07115): Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics [[code](https://github.com/ranandalon/mtl)]

## BibTex
```
@misc{gama2025uniformlossvsspecialized,
      title={Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning}, 
      author={Gabriel S. Gama and Valdir Grassi Jr},
      year={2025},
      eprint={2505.10347},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.10347}, 
}
```
