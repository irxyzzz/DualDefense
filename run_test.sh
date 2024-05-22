python main.py --num_clients 100 --dataset mnist --fusion fedavg --training_round 10 --local_epochs 3 --optimizer sgd --batch_size 64 --regularization 1e-5
python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 10 --local_epochs 3 --batch_size 64 --attacker_strategy model_poisoning_ipm --attack_start_round 5 --attacker_ratio 0.2 --epsilon 0.01
