# -----------ARGS---------------------
pretrain_train_path ="/data/none404/few/emotion_torch/data/corpus_data/256_corpus.txt" #预训练数据路径
pretrain_dev_path = ""  #MLM任务验证集数据，大多数情况选择不验证（predict需要时间,知道验证集只是表现当前MLM任务效果）

max_seq_length = 256
do_train = True
do_eval =False
do_lower_case = False #数据是否全变成小写（是否区分大小写）

train_batch_size = 64 #根据卡而定
eval_batch_size = 32
learning_rate = 1e-4
num_train_epochs = 16
save_checkpoints_steps = 15000 #保存步数
warmup_proportion = 0.1 #前warmup_proportion的步伐 慢热学习比例
dupe_factor = 1 #动态掩盖倍数
no_cuda = False #是否使用gpu
local_rank = -1  #分布式训练
seed = 42 #随机种子

gradient_accumulation_steps = 1 #梯度累积（相同显存下能跑更大的batch_size）1不使用
fp16 = False #混合精度训练
loss_scale = 0. #0时为动态
bert_config_json = "bert_config.json" #bert Transormer的参数设置
vocab_file = "vocab.txt"
init_model = 'pretraining_model' #预训练模型所在路径（文件夹）为''时从零训练，不为''时继续训练。huggingface roberta 下载链接为：https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
output_dir = "/data/none404/few/emotion_torch/pretrainning_bert/Mybert/outputs/"
masked_lm_prob = 0.15
max_predictions_per_seq = 20

frozen = True  #冻结word_embedding参数

#bert参数解释
"""
{
  "attention_probs_dropout_prob": 0.1, #乘法attention时，softmax后dropout概率 
  "directionality": "bidi", 
  "hidden_act": "gelu", # 激活函数
  "hidden_dropout_prob": 0.1, #隐藏层dropout概率
  "hidden_size": 768, # 最后输出词向量的维度
  "initializer_range": 0.02, # 初始化范围
  "intermediate_size": 3072, # 升维维度
  "max_position_embeddings": 512, # 最大的
  "num_attention_heads": 12, # 总的头数
  "num_hidden_layers": 12, #隐藏层数 ，也就是transformer的encode运行的次数
  "pooler_fc_size": 768, 
  "pooler_num_attention_heads": 12, 
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform", 
  "type_vocab_size": 2, #segment_ids类别 [0,1]
  "vocab_size": 21128 #词典中词数
}
"""
