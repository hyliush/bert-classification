import argparse
import sys
import logging
# MODEL_CLASSES = {'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)}
# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in ( BertConfig,)), ())

class Config(object):
    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser()
        ## Required parameters
        parser.add_argument("--model_type", default=None, type=str, required=True,
                            help="Model type selected in the list.")

        parser.add_argument("--data_dir", default=None, type=str, required=True,
                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

        parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                            help="Path to pre-trained model or shortcut name selected in the list: ")

        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument('--num_classification',required=True,type=int,default=3,
                            help="the number of categories")
        parser.add_argument("--num_train_epochs", default=10, type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument('--predict_text', default="", type=str, help="Predict sentiment on a given sentence")
        parser.add_argument('--predict_filename', default="", type=str, help="Predict sentiment on a given file")
        parser.add_argument("--max_seq_length", default=128, type=int,required=True,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                 "than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--evaluate_during_training", action='store_true',
                            help="Rul evaluation during training at each logging step.")
        ## Other parameters
        parser.add_argument('--task',type=int,nargs='+',
                            help="The task. for example dummy classification[0,1]")
        parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--cache_dir", default="", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--do_train", action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_test", action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_eval", action='store_true',
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--eval_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")

        parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--train_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for training.")

        parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for evaluation.")

        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--learning_rate", default=5e-5, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float,
                            help="Weight deay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--eval_steps", default=-1, type=int,
                            help="")
        parser.add_argument("--lstm_hidden_size", default=300, type=int,
                            help="")
        parser.add_argument("--lstm_layers", default=2, type=int,
                            help="")
        parser.add_argument("--lstm_dropout", default=0.5, type=float,
                            help="")
        parser.add_argument("--train_steps", default=-1, type=int,
                            help="")
        parser.add_argument("--report_steps", default=-1, type=int,
                            help="")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="Linear warmup over warmup_steps.")
        parser.add_argument("--split_num", default=3, type=int,
                            help="text split")
        parser.add_argument('--logging_steps', type=int, default=50,
                            help="Log every X updates steps.")
        parser.add_argument('--save_steps', type=int, default=50,
                            help="Save checkpoint every X updates steps.")
        parser.add_argument("--eval_all_checkpoints", action='store_true',
                            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Avoid using CUDA when available")
        parser.add_argument('--overwrite_output_dir', action='store_true',
                            help="Overwrite the content of the output directory")
        parser.add_argument('--overwrite_cache', action='store_true',
                            help="Overwrite the cached training and evaluation sets")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")

        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
        parser.add_argument('--fp16_opt_level', type=str, default='O1',
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
        parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
        parser.add_argument("--freeze", default=0, type=int, required=False,
                            help="freeze bert.")
        parser.add_argument("--not_do_eval_steps", default=0.35, type=float,
                            help="not_do_eval_steps.")
        return parser

    @classmethod
    def get_default_cofig(cls):
        sys.argv = ['name',
                    '--model_type', 'bert',
                    '--model_name_or_path', './model/chinese_roberta_wwm_ext_pytorch',
                    '--do_test',
                    '--do_train',
                    '--evaluate_during_training',
                    '--num_train_epochs', '20',
                    '--data_dir', './data/data_origin_1',
                    '--output_dir', './model/roberta_wwm_5121_125',
                    '--predict_text', '而此前为“避险”停牌的上市公司又开始谋划着闪电复牌，于是相较部分较为正经的复牌理由，一些奇葩说词成为“灾后重建”的一道风景线',
                    '--predict_filename', 'test.csv',
                    '--num_classification','2',
                    '--max_seq_length', '120',
                    '--task', '0', '1',
                    '--lstm_hidden_size', '512', '--lstm_layers', '1',
                    '--lstm_dropout', '0.1', '--eval_steps', '200',
                    '--train_batch_size', '2',
                    '--gradient_accumulation_steps', '4', '--warmup_steps', '0',
                    '--eval_batch_size', '8', '--learning_rate', '1e-5',
                    '--adam_epsilon', '1e-6',
                    '--weight_decay', '0', '--train_steps', '30000', '--freeze', '0']
        parser = cls.get_parser()
        args = parser.parse_args()
        # if args.local_rank == -1 or args.no_cuda:
        #     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #     args.n_gpu = torch.cuda.device_count()
        # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #     torch.cuda.set_device(args.local_rank)
        #     device = torch.device("cuda", args.local_rank)
        #     torch.distributed.init_process_group(backend='nccl')
        #     args.n_gpu = 1
        # args.device = device
        return  args



if __name__ == '__main__':
    config = Config()
    args = config.get_default_cofig()


