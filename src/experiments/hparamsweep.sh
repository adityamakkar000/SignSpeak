# bash script to run hyperparameter sweep experiments

# Basic setup

time_steps=79
batch_size=64
epochs=5

# General Model params

model="GRU"
hidden_size=32

# RNN params
dense_layer=false
dense_size=$(($hidden_size / 2))

if [ "$dense_layer" = true ]; then
  dense_layer_arg="-dense_layer"
else
  dense_layer_arg=""
fi

# Encoder params
number_heads=0

# Description for run
# description="GRU_Test"
project_name="GRU_Test"

for lr in 0.01 0.001 0.0001
  do
    for layers in 1 2
      do
        description="GRU_Test_lr_${lr}_layers_${layers}"
        python LightningTrain.py \
          -layers $layers \
          -model $model \
          -hidden_size $hidden_size \
          -lr $lr \
          -time_steps $time_steps \
          -batch_size $batch_size \
          -epochs $epochs \
          $dense_layer_arg \
          -dense_size $dense_size \
          -description $description \
          -project_name $project_name
    done
done
