# Bert for classification in Pytorch
## Introduce
BERT is state-of-the-art natural language processing model from Google. Using its latent space, it can be repurpossed for various NLP tasks, such as sentiment analysis.  

This simple wrapper based on Transformers (for managing BERT model) and PyTorch 
## Structure
- data folder contains files(train.csv,dev.csv) whose columns are text content and label.  

- model folder contains files (.bin,.json,.txt) can be downloaded [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) for chinese text (also support other language).

- [dataset.py](https://github.com/hyliush/bert-classification/blob/main/dataset.py)
Text process -> read a file and convert it to a format for Bert.

- [model.py](https://github.com/hyliush/bert-classification/blob/main/model.py)
Build Model  -> modules contain  model train (a simple scheme for earlystop), model predict and model evaluation. 

- [config.py](https://github.com/hyliush/bert-classification/blob/main/config.py)
Config -> One can use Config.get_default_config() to run in a IDE.

- [main.py](https://github.com/hyliush/bert-classification/blob/main/main.py)
Run a model ->1. train model  Training with default parameters can be performed simply by adding --do_train.  Must run it when first use.
  2.predict text or a file  --predict_text for a sententce prediction and --predict_filename for a file contained texts prediction.   

`python main.py --model_type bert --model_name_or_path ../model/chinese_roberta_wwm_ext_pytorch
                    --do_test
                    --do_train
                    --evaluate_during_training
                    --num_train_epochs 20
                    --data_dir './data/data_origin_1'
                    --output_dir  './model/roberta_wwm_5121_125'
                    --predict_text "你好"
                    --predict_filename test.csv
                    --num_classification 2
                    --max_seq_length 120
                    --task, 0 1
                    --lstm_hidden_size 512  --lstm_layers 1
                    --lstm_dropout 0.1
                    --eval_steps 200 --train_batch_size 2
                    --gradient_accumulation_steps 4 --warmup_steps 0
                    --eval_batch_size 8 --learning_rate 1e-5 --adam_epsilon 1e-6
                    --weight_decay, 0 --train_steps 30000 --freeze 0`
## 
