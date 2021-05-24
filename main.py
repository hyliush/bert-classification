from torch.utils.data import RandomSampler,SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig,BertTokenizer,BertForSequenceClassification
# from pytorch_transformers.modeling_bert import BertForSequenceClassification

import os
from config import Config
from model import SentimentBert
from dataset import SentimentData


def train(args):

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_classification)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)

    # prepare dataloader
    dt = SentimentData(tokenizer=tokenizer)
    if args.local_rank == -1:
        train_sampler = RandomSampler
    else:
        train_sampler = DistributedSampler

    train_dataloader = dt.prepare_dataloader(file_path=os.path.join(args.data_dir, 'train.csv'),
                                             batch_size=args.train_batch_size,
                                             max_seq_length=args.max_seq_length,
                                             sampler=train_sampler)
    # predictor
    predictor = SentimentBert(args)
    predictor.train(model, train_dataloader, tokenizer)


def predict(args,**kwargs):
    # prepare dataloader
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    dt = SentimentData(tokenizer=tokenizer)
    if kwargs.get('text'):
        dataloader = dt.prepare_dataloader_from_iterator([('uts1299034-124', kwargs['text'], 0)],
                                                         args.eval_batch_size,
                                                         args.max_seq_length,
                                                         sampler=SequentialSampler)
    if kwargs.get('filename'):
        dataloader = dt.prepare_dataloader(os.path.join(args.data_dir, kwargs['filename']),
                                            args.eval_batch_size,
                                            args.max_seq_length,
                                            sampler=SequentialSampler)
    # load model
    predictor = SentimentBert(args)
    predictor.load_model(model_dir=os.path.join(args.output_dir, "pytorch_model.bin"))

    result = predictor.predict(dataloader)

    return result

if __name__ == '__main__':
    import sys
    import pandas as pd
    sys.argv =sys.argv[:1]+\
                        ['--model_type', 'bert',
                        '--model_name_or_path', './model/chinese_roberta_wwm_ext_pytorch',
                        '--do_test',
                        #'--do_train',
                        '--evaluate_during_training',
                        '--num_train_epochs', '20',
                        '--data_dir', './data/data_origin_1',
                        '--output_dir', './model/roberta_wwm_5121_125',
                        '--predict_text', '你好',
                        '--predict_filename', 'test.csv',
                        '--num_classification', '2',
                        '--max_seq_length', '120',
                        '--train_batch_size', '2',
                        '--eval_batch_size', '8',
                        '--task', '0', '1',
                        '--lstm_hidden_size', '512', '--lstm_layers', '1',
                        '--lstm_dropout', '0.1', '--eval_steps', '200',
                        '--gradient_accumulation_steps', '4', '--warmup_steps', '0',
                        '--learning_rate', '1e-5', '--adam_epsilon', '1e-6',
                        '--weight_decay', '0', '--train_steps', '30000', '--freeze', '0']
    parser = Config.get_parser()
    args = parser.parse_args()

    # train model
    if args.do_train:
        # prepare model
        train(args)

    # predict text
    if args.do_test:
        result = predict(args=args,text = args.predict_text)
        print(result['infer_labels'])

        result = predict(args=args,filename = args.predict_filename)
        logits = result['infer_logits']

        df=pd.read_csv(os.path.join(args.data_dir, args.predict_filename))
        for i in range(args.num_classification):
            df['label_{}'.format(str(i))]=logits[:,i]
        df[['_id']+['label_{}'.format(str(i)) for i in range(args.num_classification)]].to_csv(
            os.path.join(args.output_dir, "sub.csv"),index=False)





