# optimized parameters

# starting point - same as https://github.com/yobibyte/unitary-scalarization-dmtl
CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py --optimizer baseline --dataset cityscapes --model resnet50 --tasks S_D --aug --n_classes 7 --shape 256 128 --lr 0.0005 --p 0.0 --weight_decay 1e-4 --decay_lr --batch_size 32 --num_epochs 150 --n_runs 2 --start_index 1 --analysis_test "grad_metrics"
CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py --optimizer imtl --dataset cityscapes --model resnet50 --tasks S_D --aug --n_classes 7 --shape 256 128 --lr 0.0005 --p 0.0 --weight_decay 1e-4 --decay_lr --batch_size 32 --num_epochs 150 --n_runs 2 --start_index 1 --analysis_test "grad_metrics"

# change: resnet50 -> resnet18
CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py --optimizer baseline --dataset cityscapes --model resnet18 --tasks S_D --aug --n_classes 7 --shape 256 128 --lr 0.001 --p 0.0 --weight_decay 1e-4 --decay_lr --batch_size 32 --num_epochs 150 --n_runs 2 --start_index 1 --analysis_test "grad_metrics"
CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py --optimizer imtl --dataset cityscapes --model resnet18 --tasks S_D --aug --n_classes 7 --shape 256 128 --lr 0.0005 --p 0.0 --weight_decay 1e-4 --decay_lr --batch_size 32 --num_epochs 150 --n_runs 2 --start_index 1 --analysis_test "grad_metrics"

# change: added I task
CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py --optimizer baseline --dataset cityscapes --model resnet18 --tasks S_D_I --aug --n_classes 7 --shape 256 128 --lr 0.001 --p 0.0 --weight_decay 1e-4 --decay_lr --batch_size 32 --num_epochs 150 --n_runs 2 --start_index 1 --analysis_test "grad_metrics"
CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py --optimizer imtl --dataset cityscapes --model resnet18 --tasks S_D_I --aug --n_classes 7 --shape 256 128 --lr 0.001 --p 0.0 --weight_decay 1e-4 --decay_lr --batch_size 32 --num_epochs 150 --n_runs 2 --start_index 1 --analysis_test "grad_metrics"

# change: increased resolution
CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py --optimizer baseline --dataset cityscapes --model resnet18 --tasks S_D_I --aug --n_classes 7 --shape 512 256 --lr 0.001 --p 0.0 --weight_decay 1e-4 --decay_lr --batch_size 32 --num_epochs 150 --n_runs 2 --start_index 1 --analysis_test "grad_metrics"
CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py --optimizer imtl --dataset cityscapes --model resnet18 --tasks S_D_I --aug --n_classes 7 --shape 512 256 --lr 0.001 --p 0.0 --weight_decay 1e-4 --decay_lr --batch_size 32 --num_epochs 150 --n_runs 2 --start_index 1 --analysis_test "grad_metrics"

# change: 7 classes -> 19 classes (already optimized values) - running again convenience
CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py --optimizer baseline --dataset cityscapes --model resnet18 --tasks S_D_I --aug --n_classes 19 --shape 512 256 --lr 0.001 --p 0.0 --weight_decay 1e-4 --decay_lr --batch_size 32 --num_epochs 150 --n_runs 3 --start_index 0 --analysis_test "grad_metrics"
CUDA_VISIBLE_DEVICES=0 python3.8 supervised_experiments/train_multi_task.py --optimizer imtl --dataset cityscapes --model resnet18 --tasks S_D_I --aug --n_classes 19 --shape 512 256 --lr 0.001 --p 0.0 --weight_decay 1e-4 --decay_lr --batch_size 32 --num_epochs 150 --n_runs 3 --start_index 0 --analysis_test "grad_metrics"