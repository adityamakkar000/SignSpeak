

lr=0.0001
time_steps=60
batch_size=64
epochs=600
model="LSTM"
hidden_size=64

python LightningTrain.py -model $model -hidden_size $hidden_size -lr $lr -time_steps $time_steps -batch_size $batch_size -epochs $epochs
