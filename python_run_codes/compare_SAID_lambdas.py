import torch


meta_said_file = 'META_SAID_global_params_intrinsic_dim_600'
ella_said_file = 'ELLA_SAID_META/global_params/intrinsic_dim_600'

meta_said_dict = torch.load(meta_said_file)
ella_said_dict = torch.load(ella_said_file)

print("==============================meta SAID==============================")
for key, value in meta_said_dict.items():
    print(key+':')
    print(value.detach().cpu().numpy())
    print(".........................................")

print("==============================ELLA SAID==============================")
for key, value in ella_said_dict.items():
    print(key+':')
    print(value.detach().cpu().numpy())
    print(".........................................")
    
