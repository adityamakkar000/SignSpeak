
lr=0.0001
time_steps=79
batch_size=64
epochs=1000
model="Encoder"
hidden_size=32
layers=1
description="encoder_higherDim"

python LightningTrain.py \
  -layers $layers \
  -model $model \
  -hidden_size $hidden_size \
  -lr $lr \
  -time_steps $time_steps \
  -batch_size $batch_size \
  -epochs $epochs \
  # -description $description # Uncomment to save model with description
