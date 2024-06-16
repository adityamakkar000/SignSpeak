# Basic setup
lr=0.001
time_steps=79
batch_size=64
epochs=500

# General Model params

hidden_size=32

# RNN params

# Encoder params
number_heads=1

# Description for run

for model in "Encoder"; do
  for l in 1 2 3 4 5; do
    for number_heads in 1 2 4 8; do

      layers=$l

      description="Encoder_sweep_aarav"

      python LightningTrain.py \
        -layers $layers \
        -model $model \
        -hidden_size $hidden_size \
        -lr $lr \
        -time_steps $time_steps \
        -batch_size $batch_size \
        -epochs $epochs \
        -number_heads $number_heads \
        # -description $description # Uncomment to save model with description

    done
  done
done

# layers=$l

#     # if [ "$dense_layer" = true ]; then
#     #   dense_layer_arg="-dense_layer"
#     # else
#     #   dense_layer_arg=""
#     # fi

#     description="EncoderfixedLR_Sweep"

#     python LightningTrain.py \
#       -layers $layers \
#       -model $model \
#       -hidden_size $hidden_size \
#       -lr $lr \
#       -time_steps $time_steps \
#       -batch_size $batch_size \
#       -epochs $epochs \
#       -number_heads $number_heads \
#       # -description $description # Uncomment to save model with description
