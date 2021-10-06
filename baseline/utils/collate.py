import torch

""" 
    Custom collate function.
    We need to reorder the batch based on lengths of the sentences. 
    Since pytorch requires an ordered tensor for the lstm.
    lengths, sort_id = torch.Tensor([x for x in lengths]).sort(descending=True)
"""
def custom_collate(batch):
    output = {k: [] for k in batch[0].keys()}
    
    # Group all values together as a list with the corresponding key 
    for sample in batch:
        for k in output.keys():
            output[k].append(sample[k])
    
    output['command_length'] = torch.LongTensor([c for c in output['command_length']])

    # Sort the samples bases on command length
    lengths, sort_id = output['command_length'].sort(descending=True)
    sort_id = sort_id.tolist()
    
    # Order all elements accordingly
    output = {k: [v[i] for i in sort_id] for k, v in output.items()} 

    # Stack as tensors
    output = {k: torch.stack(v, 0).squeeze() for k, v in output.items()}    

    return output 
