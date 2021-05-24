from transformers import BertConfig, BertTokenizer,BertForSequenceClassification
# from pytorch_transformers.modeling_bert import BertForSequenceClassification
import random
import numpy as np
import torch
import os
import sys
from itertools import cycle
from tqdm import tqdm,trange
from dataset import SentimentData
from config import Config
import time
from torch.utils.data import RandomSampler,SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW,get_linear_schedule_with_warmup
from sklearn.metrics import classification_report,f1_score
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentBert(object):

    def __init__(self,args):
        self.best_score = 0.0
        self.global_step = 0  # count(gradient updated)
        self.flag = 0  # count(eval_data performance is better self.best_score)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        self.output_model_file = os.path.join(self.args.output_dir, "pytorch_model.bin")
        self.model = None

    def train(self,model,dataloader,tokenizer):
        '''
        model: Bertmodel or none(when been loaded in self.model)
        tokenizer: for the earlystop, need to load eval_data.
        '''
        if self.model is None:
            model.to(self.device)
            self.model = model
            logger.info('fine-tune based on the downloaded BertModel.')
        else:
            logger.warning('fine-tune based the loaded(trained) BertModel.')
        self._set_seed()

        # Prepare optimizer
        num_train_optimization_steps = int(len(dataloader) / self.args.gradient_accumulation_steps * self.args.num_train_epochs)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                         num_training_steps=num_train_optimization_steps)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(dataloader))
        logger.info("  Batch size = %d", self.args.train_batch_size)
        logger.info("  Total Num steps = %d", num_train_optimization_steps)

        self.model.train()
        tr_loss = 0
        dataloader = cycle(dataloader)
        bar = tqdm(range(num_train_optimization_steps*self.args.gradient_accumulation_steps), total=num_train_optimization_steps*self.args.gradient_accumulation_steps)
        for step in bar:
            batch = next(dataloader)
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss, _ = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)

            del input_ids, input_mask, segment_ids, label_ids
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if self.args.fp16 and self.args.loss_scale != 1.0:
                loss = loss * self.args.loss_scale
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            tr_loss += loss.item()
            batch_train_loss=round(tr_loss*self.args.gradient_accumulation_steps/(step+1),4)
            bar.set_description("loss {}".format(batch_train_loss))

            if self.args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            # 梯度更新
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.fp16:
                    if self.args.loss_scale != 1.0:
                        # scale down gradients for fp16 training
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad.data = param.grad.data / self.args.loss_scale
                    is_nan = self._set_optimizer_params_grad(param_optimizer, model.named_parameters(),
                                                             test_nan=True)
                    if is_nan:
                        logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                        self.args.loss_scale = self.args.loss_scale / 2
                        optimizer.zero_grad()
                        continue
                    optimizer.step()
                    scheduler.step()
                    self._copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                else:
                    optimizer.step()
                    scheduler.step()

                optimizer.zero_grad()
                self.global_step += 1
                logger.info("  %s = %s,  %s = %s", 'global_step', str(self.global_step),
                            'batch_train_loss loss',str(batch_train_loss))


                # evaluation
                if self.args.evaluate_during_training and (self.global_step + 1) % (self.args.eval_steps) == 0:
                    self._eval_data_performance(tokenizer)
                    self.model.train()
                    if self.flag > 6:
                        print('========= earlystop ==========')
                        break

    def _eval_data_performance(self,tokenizer):
        dt = SentimentData(tokenizer=tokenizer)
        eval_dataloader = dt.prepare_dataloader(file_path=os.path.join(self.args.data_dir, 'dev.csv'),
                                                batch_size=self.args.eval_batch_size,
                                                max_seq_length=self.args.max_seq_length,
                                                sampler=SequentialSampler)
        # eval_data result
        result = self._predict(eval_dataloader)
        eval_score = self.evaluate(result['gold_labels'], result['infer_labels'])
        # save model
        output_eval_file = os.path.join(self.args.output_dir, 'eval_results.txt')
        logger.info("***** Running evaluation *****  %s = %s,%s = %s ", 'eval_loss',
                    result['batch_eval_loss'], 'f1',
                    eval_score)
        with open(output_eval_file, "a") as writer:
            if (self.global_step + 1) // (self.args.eval_steps) == 1:
                writer.write(','.join(sys.argv) + '\n')
            writer.write(
                "%s, %s = %s, %s = %s\n" % (time.strftime('%Y/%m/%d %H:%M:%S', time.localtime()),
                                            'eval_loss', result['batch_eval_loss'], 'f1', eval_score))
            writer.write('*' * 80 + '\n')
        if eval_score > self.best_score:
            self.flag = 0
            self.best_score = eval_score
            print("****** Best F1({}) update ****".format(eval_score))
            print("Saving Model......")
            # Save a trained model
            model_to_save = self.model.module if hasattr(self.model,
                                                    'module') else self.model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), self.output_model_file)
            print("Model has been saved......")
        else:
            # earlystop
            self.flag += 1

    def _set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.device == 'gpu':
            torch.cuda.manual_seed_all(self.args.seed)

    def _predict(self, dataloader):
        '''
        return:
        result = {'batch_eval_loss': loss / nb_steps,
                  'infer_logits': inference_logits,
                  'infer_labels': inference_labels,
                  'gold_labels': gold_labels
                  }
        '''
        inference_labels = []
        gold_labels = []
        inference_logits = []

        self.model.eval()
        loss,nb_steps = 0, 0
        for batch in tqdm(dataloader,desc='Predicting',total=len(dataloader),nrows=1):
            batch = tuple(i.to(self.device) for i in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                tmp_loss, logits = self.model(input_ids=input_ids, token_type_ids=segment_ids,
                                              attention_mask=input_mask,
                                              labels=label_ids)
            # 记录损失和步次
            loss += tmp_loss.mean().item()
            nb_steps += 1

            inference_labels.append(np.argmax(logits.detach().cpu().numpy(), axis=1))
            gold_labels.append(label_ids.to('cpu').numpy())
            inference_logits.append(logits.detach().cpu().numpy())

        inference_labels = np.concatenate(inference_labels, 0)
        gold_labels = np.concatenate(gold_labels, 0)
        inference_logits = np.concatenate(inference_logits, 0)

        result = {'batch_eval_loss': loss / nb_steps,
                  'infer_logits': inference_logits,
                  'infer_labels': inference_labels,
                  'gold_labels': gold_labels
                  }
        return result


    def load_model(self, model_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError("folder `{}` does not exist. Please make sure model are there.".format(model_dir))
        config = BertConfig.from_pretrained(self.args.model_name_or_path, num_labels=self.args.num_classification)
        self.model = BertForSequenceClassification.from_pretrained(model_dir,config=config)
        self.model.to(self.device)

    def predict(self, dataloader):
        if self.model is None:
            raise FileNotFoundError("model not been loaded.")

        result = self._predict(dataloader)
        return result

    @staticmethod
    def evaluate(y_true,y_pred,indicator='f1-score'):
        report = classification_report(y_true, y_pred, output_dict=True)
        score = report['macro avg'][indicator]
        return score

    def _set_optimizer_params_grad(self,named_params_optimizer, named_params_model, test_nan=False):
        """ Utility function for optimize_on_cpu and 16-bits training.
            Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
        """
        is_nan = False
        for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
            if name_opti != name_model:
                logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
                raise ValueError
            if param_model.grad is not None:
                if test_nan and torch.isnan(param_model.grad).sum() > 0:
                    is_nan = True
                if param_opti.grad is None:
                    param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
                param_opti.grad.data.copy_(param_model.grad.data)
            else:
                param_opti.grad = None
        return is_nan

    def _copy_optimizer_params_to_model(self,named_params_model, named_params_optimizer):
        """ Utility function for optimize_on_cpu and 16-bits training.
            Copy the parameters optimized on CPU/RAM back to the model on GPU
        """
        for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
            if name_opti != name_model:
                logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
                raise ValueError
            param_model.data.copy_(param_opti.data)

if __name__ =='__main__':
    config = Config()
    args = config.get_default_cofig()
    # prepare model
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_classification)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config = config)

    # prepare dataloader
    dt = SentimentData(tokenizer=tokenizer)
    if args.local_rank == -1:
        train_sampler = RandomSampler
    else:
        train_sampler = DistributedSampler

    train_dataloader = dt.prepare_dataloader(file_path=os.path.join(args.data_dir, 'train.csv'),
                                             batch_size=args.train_batch_size,
                                             max_seq_length=args.max_seq_length, sampler=train_sampler)
    # predictor
    predictor = SentimentBert(args)
    predictor.train(model,train_dataloader,tokenizer)