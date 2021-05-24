# The Predictive Power of Investor Sentiment to Copper Futures Volatility in China -- Based on DMA-HAR-CJ Model

This article builds a news database by crawling the historical news of China Financial News Network from 2007 to the present, and uses the Bertmodel to extract news sentiment as a proxy variable for investor sentiment, and uses the N-gram model to calculate news crossentropy as a market unusualness. Taking into account the uncertainty of the impact of investor sentiment on the volatility of China's copper futures market and the jumping phenomenon of the copper futures market, this article uses the DMA-HAR-CJ model as the benchmark model for volatility prediction. By incorporating investor sentiment variables into the benchmark model, Constructed the sentiment model DMA-HAR-CJ-Sen to study, the in-sample and out-of-sample prediction ability of investor sentimen, especially abnormal investor sentiment on the realized volatility of China's copper futures market.  

the main module contains four parts:
## The first part: Crawing News from websites
### Structure
- crawler folder contains main script for crawling from [中国金融界](http://www.jrj.com.cn),[中国证券网](https://www.cnstock.com) and [每经网](http://www.nbd.com.cn). The script
is revised based on [DemonDamon](https://github.com/DemonDamon/Listed-company-news-crawl-and-text-analysis). And especially, we add a script for crawling eastmoney forum.
- ProxyPool folder for crawling proxy. More details can been seen []. 
- config,database,dedupication,denull,tokenzition,and utils are used to do some data processing.


## The sencond part: Bert for sentiment classification as a proxy of investor sentiment.
### Introduce
BERT is state-of-the-art natural language processing model from Google. Using its latent space, it can be repurpossed for various NLP tasks, such as sentiment analysis.  

This simple wrapper based on Transformers (for managing BERT model) and PyTorch 
### Structure
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
### other 
- 1.one can add some trick to import prediction performance. For example model average,Pseudo label,model stacking. Details can be seen[BDCI top1 scheme](https://github.com/cxy229/BDCI2019-SENTIMENT-CLASSIFICATION).
- 2.other deep network model can be added in model.py not only Bert class models.

## The three part: N-grams for calculate News crossentropy as a proxy of market unusualness

## The four part: DMA-HAR-CJ-Sent for volatility modeling

