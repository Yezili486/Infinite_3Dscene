#!/usr/bin/env python3
"""
Test different network configurations for Hugging Face downloads
"""

import os
import requests
import urllib3
from huggingface_hub import HfApi, snapshot_download

def test_network_configs():
    """Test different network configurations"""
    
    # Set environment variables
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '0'
    
    # Test different configurations
    configs = [
        {
            'name': 'Default',
            'timeout': 30,
            'max_retries': 3
        },
        {
            'name': 'Long timeout',
            'timeout': 120,
            'max_retries': 5
        },
        {
            'name': 'Conservative',
            'timeout': 60,
            'max_retries': 10
        }
    ]
    
    for config in configs:
        print(f"\n=== Testing {config['name']} configuration ===")
        
        try:
            # Configure session
            session = requests.Session()
            session.timeout = config['timeout']
            
            # Test basic connection
            print(f"Testing connection to huggingface.co...")
            response = session.get('https://huggingface.co/api/models/runwayml/stable-diffusion-inpainting', timeout=config['timeout'])
            print(f"✅ Connection successful: {response.status_code}")
            
            # Try to download a small file
            print(f"Testing small file download...")
            api = HfApi()
            
            # Try to get model info
            model_info = api.model_info("runwayml/stable-diffusion-inpainting")
            print(f"✅ Model info retrieved: {model_info.modelId}")
            
            return True
            
        except Exception as e:
            print(f"❌ {config['name']} failed: {e}")
    
    return False

if __name__ == "__main__":
    print("Testing network configurations for Hugging Face...")
    success = test_network_configs()
    
    if success:
        print("\n✅ Network test successful! You can try downloading the model now.")
    else:
        print("\n❌ All network configurations failed. Consider using a VPN or different network.") 