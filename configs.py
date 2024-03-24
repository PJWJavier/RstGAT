def GLUE_opts(parser):
    parser.add_argument('--pretrained_model', default='dataset/BART/pytorch_model.bin', type=str, help='')
    parser.add_argument('--task_name', default='CoLA', type=str, help='GLUE_tasks')
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=2, type=int)  # try 16, 32, 64 for BERT models
    parser.add_argument('--log_step', default=400, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=1024, type=int)
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--save_dic', default=1, type=int)
    parser.add_argument('--diff_lr', default=0.005, type=float)
    parser.add_argument('--optimizer', default='bertAdam', type=str)
    parser.add_argument("--config_path", type=str, default="dataset/BART/config.json",
                        help="Config file of model hyper-parameters.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="None",
                        help="Pooling type.")
    parser.add_argument("--cxg_vocab_path",default="dataset/CxGBERT",
                        help="Pooling type.")
    tokenizer_opts(parser)
    model_opts(parser)

def tokenizer_opts(parser):
    parser.add_argument("--tokenizer", choices=["bert", "cxgbert"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer."
                             "CxG tokenizer(Ours)"
                             )
    parser.add_argument("--word_vocab_path", default='dataset/BERT/', type=str, help="Path of the vocabulary file.")
    parser.add_argument("--do_lower_case", default=True, type=bool, help="Lowercase Characters.")
    # TODO : [word_vocab_path] Settings

def model_opts(parser):
    parser.add_argument("--embedding", choices=["bert"], default="bert", help="Emebdding type.")
    parser.add_argument("--encoder", choices=["transformer"], default="transformer", help="Encoder type.")
    parser.add_argument("--target", choices=["bert"], default="bert", help="Target type.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length for word embedding.")

    parser.add_argument("--relative_position_embedding", action="store_true",
                        help="Use relative position embedding.")
    parser.add_argument("--relative_attention_buckets_num", type=int, default=32,
                        help="Buckets num of relative position embedding.")
    parser.add_argument("--remove_embedding_layernorm", action="store_true",
                        help="Remove layernorm on embedding.")
    parser.add_argument("--remove_attention_scale", action="store_true",
                        help="Remove attention scale.")
    parser.add_argument("--mask", choices=["fully_visible", "causal", "causal_with_prefix"], default="fully_visible",
                        help="Mask type.")
    parser.add_argument("--layernorm_positioning", choices=["pre", "post"], default="post",
                        help="Layernorm positioning.")
    parser.add_argument("--feed_forward", choices=["dense", "gated"], default="dense",
                        help="Feed forward type, specific to transformer model.")
    parser.add_argument("--remove_transformer_bias", action="store_true",
                        help="Remove bias on transformer layers.")
    parser.add_argument("--layernorm", choices=["normal", "t5"], default="normal",
                        help="Layernorm type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true", help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")
    parser.add_argument("--has_residual_attention", action="store_true", help="Add residual attention.")


def optimization_opts(parser):
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--fp16_opt_level", choices=["O0", "O1", "O2", "O3" ], default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"],
                        default="adamw",
                        help="Optimizer type.")
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                                "constant", "constant_with_warmup"],
                        default="linear", help="Scheduler type.")


def training_opts(parser):
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

def deepspeed_opts(parser):
    parser.add_argument("--deepspeed", action="store_true",
                        help=".")
    parser.add_argument("--deepspeed_config", default="dataset/BERT/deepspeed_config.json", type=str,
                        help=".")
    # TODO : [deepspedd_config] Settings
    parser.add_argument("--local_rank", type=int, required=False)