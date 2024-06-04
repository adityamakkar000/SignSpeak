# Basic setup
lr=0.01
time_steps=79
batch_size=64
epochs=1000

# General Model params

model="GRU"
hidden_size=32
layers=1

# RNN params
dense_layer=true
dense_size=$(($hidden_size / 2))

if [ "$dense_layer" = true ]; then
  dense_layer_arg="-dense_layer"
else
  dense_layer_arg=""
fi

# Encoder params
number_heads=0

# Description for run
description="GRU_Test"


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
  -description $description # Uncomment to save model with description
