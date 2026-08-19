import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pisl.models import vit_base_part

model = vit_base_part(num_parts=3, num_classes=3000)
model.eval()
x = torch.randn(2, 3, 384, 128)
f_g, f_p = model.extract_all_features(x)
print(f"f_g shape: {f_g.shape}")
print(f"f_p shape: {f_p.shape}")
