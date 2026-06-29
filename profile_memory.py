import torch
import sys
import os

# Add the project directory to path so we can import pisl
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pisl.models import resnet50part

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(num):
    return f"{num / 1e6:.2f} M"

def format_memory(bytes):
    return f"{bytes / (1024**3):.2f} GB"

def main():
    print("="*60)
    print("PISL Model Complexity & Memory Profiler")
    print("="*60)
    
    # 1. Parameter Analysis
    print("\n[1] Parameter Breakdown")
    print("-" * 40)
    model = resnet50part(num_parts=3, num_classes=3000)
    
    # Break down components
    backbone_params = count_parameters(model.base) + count_parameters(model.gap) + count_parameters(model.bnneck)
    patch_gen_params = count_parameters(model.patch_proposal)
    diffusion_params = count_parameters(model.diffusion_patch)
    
    # Count classifiers
    classifier_params = count_parameters(model.classifier)
    for i in range(3): # num_parts = 3
        classifier_params += count_parameters(getattr(model, f'classifier{i}'))
        classifier_params += count_parameters(getattr(model, f'bnneck{i}'))

    total_params = count_parameters(model)
    
    print(f"{'ResNet-50 Backbone:':<30} {format_params(backbone_params):>10}")
    print(f"{'Spatial Transformer (STN):':<30} {format_params(patch_gen_params):>10}")
    print(f"{'Diffusion Module (SDM):':<30} {format_params(diffusion_params):>10}")
    print(f"{'Classifier Heads (3000 IDs):':<30} {format_params(classifier_params):>10}")
    print("-" * 40)
    print(f"{'TOTAL PARAMETERS:':<30} {format_params(total_params):>10}")
    
    # 2. VRAM Analysis
    print("\n[2] Empirical GPU Memory Analysis")
    print("-" * 40)
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine. Cannot perform VRAM profile.")
        return
        
    print("Initializing model on GPU...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = model.cuda()
    model.train()
    
    mem_model = torch.cuda.memory_allocated()
    print(f"{'Model Weights (FP32) VRAM:':<35} {format_memory(mem_model):>10}")
    
    # Default training settings from the paper
    batch_size = 64
    height = 384
    width = 128
    
    print(f"\nSimulating Training Forward/Backward Pass...")
    print(f"Batch Size: {batch_size}")
    print(f"Image Resolution: {height}x{width}")
    
    try:
        # Dummy input
        dummy_input = torch.randn(batch_size, 3, height, width).cuda()
        
        # Forward pass
        outputs = model(dummy_input)
        
        mem_forward = torch.cuda.max_memory_allocated()
        print(f"{'Peak VRAM after Forward Pass:':<35} {format_memory(mem_forward):>10}")
        
        # Backward pass simulation
        loss = outputs[2].sum() + outputs[3].sum() + outputs[4].sum()
        loss.backward()
        
        mem_backward = torch.cuda.max_memory_allocated()
        print(f"{'Peak VRAM during Backward Pass:':<35} {format_memory(mem_backward):>10}")
        
        print("\n[NOTE] The memory above DOES NOT include:")
        print("- 4 CameraContrast Memory Banks (~100-200MB)")
        print("- FAISS Jaccard distance matrix (~1-2GB)")
        print("- PyTorch CUDA Context Overhead (~1GB)")
        print("- Adam Optimizer Momentum States (~1GB)")
        print("\n=> Total required VRAM will be roughly ~4GB higher than the Peak VRAM shown above.")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n[!] OUT OF MEMORY ERROR (OOM) [!]")
            print(f"Failed at batch size {batch_size}. Your current GPU cannot fit the activations.")
            print(f"Peak VRAM before crash: {format_memory(torch.cuda.max_memory_allocated())}")
            print("\nThis crash log is perfect proof that more GPU VRAM is required.")
        else:
            raise e

if __name__ == '__main__':
    main()
