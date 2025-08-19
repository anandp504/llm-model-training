#!/usr/bin/env python3
"""
GPU Diagnostic Script for Linux NVIDIA Setup
Run this to check why autotrain can't detect your GPU
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, description):
    """Run a command and return the result"""
    print(f"\n=== {description} ===")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Success:")
            print(result.stdout)
        else:
            print("❌ Failed:")
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_system_info():
    """Check basic system information"""
    print("=== System Information ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Check if running in virtual environment
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        print(f"Virtual Environment: {venv}")
    else:
        print("Virtual Environment: Not detected")

def check_nvidia_drivers():
    """Check NVIDIA driver installation"""
    print("\n=== NVIDIA Driver Check ===")
    
    # Check nvidia-smi
    nvidia_smi_ok = run_command("nvidia-smi", "NVIDIA-SMI Check")
    
    # Check driver version
    if nvidia_smi_ok:
        run_command("cat /proc/driver/nvidia/version", "NVIDIA Driver Version")
    
    # Check if NVIDIA modules are loaded
    run_command("lsmod | grep nvidia", "NVIDIA Kernel Modules")
    
    # Check CUDA installation
    cuda_ok = run_command("nvcc --version", "CUDA Compiler Check")
    
    return nvidia_smi_ok and cuda_ok

def check_pytorch_cuda():
    """Check PyTorch CUDA installation"""
    print("\n=== PyTorch CUDA Check ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test CUDA tensor creation
            try:
                x = torch.randn(3, 3).cuda()
                print("✅ CUDA tensor creation successful")
            except Exception as e:
                print(f"❌ CUDA tensor creation failed: {e}")
        else:
            print("❌ PyTorch CUDA not available")
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    return torch.cuda.is_available()

def check_environment_variables():
    """Check relevant environment variables"""
    print("\n=== Environment Variables ===")
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_HOME',
        'LD_LIBRARY_PATH',
        'PATH'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def check_autotrain_installation():
    """Check autotrain installation"""
    print("\n=== AutoTrain Installation Check ===")
    
    try:
        import autotrain
        print(f"✅ AutoTrain version: {autotrain.__version__}")
    except ImportError:
        print("❌ AutoTrain not installed")
        return False
    
    # Check if autotrain can see CUDA
    try:
        from autotrain.trainers.common import get_device
        device = get_device()
        print(f"AutoTrain detected device: {device}")
    except Exception as e:
        print(f"❌ AutoTrain device detection failed: {e}")
    
    return True

def main():
    """Main diagnostic function"""
    print("🔍 GPU Diagnostic Tool for Linux NVIDIA Setup")
    print("=" * 50)
    
    check_system_info()
    nvidia_ok = check_nvidia_drivers()
    pytorch_ok = check_pytorch_cuda()
    check_environment_variables()
    autotrain_ok = check_autotrain_installation()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    if nvidia_ok:
        print("✅ NVIDIA drivers and CUDA are properly installed")
    else:
        print("❌ NVIDIA drivers or CUDA have issues")
    
    if pytorch_ok:
        print("✅ PyTorch can detect and use CUDA GPU")
    else:
        print("❌ PyTorch CUDA is not working")
    
    if autotrain_ok:
        print("✅ AutoTrain is properly installed")
    else:
        print("❌ AutoTrain installation has issues")
    
    if nvidia_ok and pytorch_ok and autotrain_ok:
        print("\n🎉 All checks passed! Your GPU should work with AutoTrain.")
    else:
        print("\n⚠️  Some issues detected. Check the output above for details.")
        
        if not nvidia_ok:
            print("\n💡 To fix NVIDIA driver issues:")
            print("1. Install NVIDIA drivers: sudo apt install nvidia-driver-xxx")
            print("2. Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit")
            print("3. Reboot the system")
        
        if not pytorch_ok:
            print("\n💡 To fix PyTorch CUDA issues:")
            print("1. Reinstall PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("2. Make sure CUDA version matches PyTorch requirements")
        
        if not autotrain_ok:
            print("\n💡 To fix AutoTrain issues:")
            print("1. Reinstall AutoTrain: pip install autotrain-advanced")
            print("2. Check if running in correct virtual environment")

if __name__ == "__main__":
    main()
