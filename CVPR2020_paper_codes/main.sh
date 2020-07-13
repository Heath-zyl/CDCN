export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python train_CDCN.py \
    --lr 0.0001 \
    --batchsize 64 \
    --lr_drop_step '8 16' \
    --gamma 0.5 \
    --echo_batches 10 \
    --epochs 1400 \
    --log CDCNpp_P1 \
    --exp_name recopy_02 \
    --start_epochs 5 \
    --epoch_test 5 \
    --is_load_model \
    --model_path "/ssd/ylzhang/code/CDCN/CVPR2020_paper_codes/exp/recopy_01/CDCNpp_P1_5.pkl"
