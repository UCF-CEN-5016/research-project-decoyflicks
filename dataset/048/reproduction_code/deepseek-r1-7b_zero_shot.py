generate_out = model.generate(audio_chunk)[0][0]
emissions.append(generate_out)

# Previously:
# emissions_arr = [generate_out.unsqueeze(0)]
# Now: simply collect without adding an extra dimension

if len(emissions_arr) > 1 and all(e.size(2) == emissions_arr[0].size(2) for e in emissions_arr):
    # Assuming we want to concatenate along dim=2 (features)
    aligned_emissions = torch.cat(emissions_arr, dim=2).long()
else:
    aligned_emissions = torch.stack(emissions_arr, dim=0).long()

# The rest of the code proceeds as before
if isinstance(target, Variable):
    target = target.view(-1)
target = F.softmax(torch.cat([aligned_emissions, target.unsqueeze(0)], dim=1), dim=1)