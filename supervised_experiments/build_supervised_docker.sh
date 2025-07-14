echo "${WANDB_API_KEY}"
docker build --build-arg uid=$UID --build-arg user=$USER --build-arg wandb_api_key="${WANDB_API_KEY}" -t smto_analysis ./supervised_experiments/
