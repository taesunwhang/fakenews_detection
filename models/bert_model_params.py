from models.graphs import bert_base_fine_tuning
from collections import defaultdict

BASE_PARAMS = defaultdict(
	# lambda: None,  # Set default value to None.

	# GPU params
	gpu_num = [0],

	# Input params
	train_batch_size=16,
	eval_batch_size=200,
	max_seq_length=320,

	# Training params
	learning_rate=0.00001,
	loss_type="softmax",
	less_data_rate=1.0,
	training_shuffle_num=50,
	dropout_prob=0.1,
	num_epochs=5,
	tensorboard_step=100,
	embedding_dim=768,

	# Training setup params
	do_lower_case=True,
	do_train_bert=True,
	is_train_continue=False,
	on_training=False,
	do_evaluate=False,
	do_transformer_residual=False,
	do_adam_weight_optimizer=False,
	use_one_hot_embeddings=False,

	# Train Model Config
	task_name="fnc",
	do_dialog_state_embedding=False,
	do_adapter_layer=False,
	graph=bert_base_fine_tuning,

	# Need to change to train...(e.g.data dir, config dir, vocab dir, etc.)
	init_checkpoint="/mnt/raid5/shared/bert/uncased_L-12_H-768_A-12/bert_model.ckpt",
	bert_config_dir="/mnt/raid5/shared/bert/uncased_L-12_H-768_A-12/bert_config.json",
	vocab_dir="/mnt/raid5/shared/bert/uncased_L-12_H-768_A-12/vocab.txt",
	root_dir="/mnt/raid5/taesun/bert/fnc/runs/",
	data_dir="fnc-1",

	# Others
	train_transformer_layer=range(0,11),
	evaluate_step=500,
	save_step=3000,
)