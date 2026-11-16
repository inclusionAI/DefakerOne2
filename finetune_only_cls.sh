# ---运行命令--- 
# bash finetune_only_cls.sh [main_node_ip] [this_node_index]

MASTER_IP=$1
NODE_RANK=$2

# 参数校验
if [ -z "$MASTER_IP" ] || [ -z "$NODE_RANK" ]; then
    echo "错误: 缺少参数"
    echo "用法: bash $0 <MASTER_IP> <NODE_RANK>"
    echo "示例: bash $0 192.168.1.100 0"
    exit 1
fi

# ======================== 集群配置 ========================
MASTER_PORT=29500
NNODES=1          # 总节点数，按实际修改
NPROC_PER_NODE=8 # 每节点 GPU 数

# ======================== 环境变量 ========================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MIN_PIXELS=262144
export MAX_PIXELS=2408448
export FPS_MAX_FRAMES=4
export VIDEO_MAX_TOKEN_NUM=256

# 分布式通信环境变量
export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=$MASTER_PORT
export NNODES=$NNODES
export NODE_RANK=$NODE_RANK
export NPROC_PER_NODE=$NPROC_PER_NODE

echo "===== 启动信息 ====="
echo "主节点 IP   : $MASTER_IP"
echo "当前节点序号: $NODE_RANK / $((NNODES-1))"
echo "总节点数    : $NNODES"
echo "每节点 GPU  : $NPROC_PER_NODE"
echo "===================="

# ======================== 启动训练 ========================
swift sft \
    --tuner_type lora \
    --freeze_vit False \
    --freeze_aligner False \
    --lora_rank 64 \
    --lora_alpha 128 \
    --model [pretrained_model] \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size [batch_size] \
    --split_dataset_ratio 0.001 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --output_dir [your_output_dir] \
    --learning_rate 1e-4 \
    --dataset [your_dataset]
