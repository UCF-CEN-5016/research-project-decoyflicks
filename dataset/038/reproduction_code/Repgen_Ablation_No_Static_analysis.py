import torch
from labml import add_global_step, add_pytorch_models, create, enum, get_data_path, log, loop, new_line, save, save_checkpoint, start
from labml_helpers.TextFileDataset import TextFileDataset
from labml_nn.Dataset import Dataset
from labml_nn.NearestNeighborEncoder import NearestNeighborEncoder
from labml_nn.Noam import Noam
from labml_nn.RetroIndex import RetroIndex
from labml_nn.RetroModel import RetroModel

chunk_len = 16
d_model = 128
d_ff = 512
n_heads = 16
d_k = 16
batch_size = 4

nearest_neighbor_encoder = NearestNeighborEncoder(chunk_len=chunk_len, num_chunks=3, d_model=d_model, n_heads=n_heads, d_k=d_k, d_ff=d_ff)

model = RetroModel(vocab_size=len(tds.n_tokens), d_model=d_model, num_layers=6, hidden_sizes=[128, 256], chunk_len=chunk_len, n_heads=n_heads, d_k=d_k, d_ff=d_ff, encoder=nearest_neighbor_encoder)
model.to('cuda:0')

train_ds = TextFileDataset(path=get_data_path('tiny_shakespeare.txt'), load_text_file_as_list=True)
retro_train_dataset = json.load(open(get_data_path('retro_train_dataset.json')))

train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)

noam_optimizer = Noam(model.parameters(), lr=1., d_model=d_model, warmup=2000)

trainer = create(device='cuda:0', model=model, dataloader=train_dl, optimizer=noam_optimizer)

sampler = Sampler(device='cuda:0', model=model, tds=tds, chunk_len=chunk_len)

prompt = 'Second Citizen:\\nOne word, good citizens.\\n\\nFirst Citizen:'

start()

for epoch in loop(epochs=32):
    output = trainer.train_step()
    
    x_rope_shape = output['x_rope'].shape
    cos_cached_shape = output['cos_cached'].shape
    sin_cached_shape = output['sin_cached'].shape
    
    if x_rope_shape[3] != cos_cached_shape[3]:
        raise RuntimeError(f'The size of tensor a ({x_rope_shape[3]}) must match the size of tensor b ({cos_cached_shape[3]}) at non-singleton dimension 3')

new_line()