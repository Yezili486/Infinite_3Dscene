"""
Configuration management for Close-up-GS
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration class for Close-up-GS"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration"""
        # Default configuration
        self.defaults = {
            # Model parameters
            'max_gaussians': 100000,
            'feature_dim': 32,
            'sh_degree': 3,
            'closeup_threshold': 2.0,
            
            # Training parameters
            'learning_rate': 1e-3,
            'batch_size': 1,
            'max_epochs': 100,
            'warmup_epochs': 10,
            'save_interval': 10,
            
            # Optimization parameters
            'position_lr_init': 0.00016,
            'position_lr_final': 0.0000016,
            'position_lr_delay_mult': 0.01,
            'position_lr_max_steps': 30000,
            'feature_lr': 0.0025,
            'opacity_lr': 0.05,
            'scaling_lr': 0.005,
            'rotation_lr': 0.001,
            
            # Densification parameters
            'densification_interval': 100,
            'opacity_reset_interval': 3000,
            'densify_from_iter': 500,
            'densify_until_iter': 15000,
            'densify_grad_threshold': 0.0002,
            'min_opacity': 0.005,
            'percent_dense': 0.01,
            
            # Rendering parameters
            'white_background': True,
            'debug': False,
            
            # Data parameters
            'data_device': 'cuda',
            'eval': False,
            'resolution': -1,
            'images': 'images',
            'resolution_scales': [1.0],
            
            # Close-up specific parameters
            'detail_enhancement': True,
            'adaptive_density': True,
            'super_resolution': False,
            'sr_factor': 2,
        }
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
        else:
            self._config = self.defaults.copy()
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            print(f"Config file {config_path} not found, using defaults")
            self._config = self.defaults.copy()
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            # Start with defaults and update with loaded config
            self._config = self.defaults.copy()
            if yaml_config:
                self._update_nested_config(self._config, yaml_config)
            
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            self._config = self.defaults.copy()
    
    def _update_nested_config(self, config: Dict, updates: Dict):
        """Recursively update nested configuration"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                self._update_nested_config(config[key], value)
            else:
                config[key] = value
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments"""
        # Map command line arguments to config keys
        arg_mapping = {
            'lr': 'learning_rate',
            'epochs': 'max_epochs',
            'batch_size': 'batch_size',
        }
        
        for arg_name, config_key in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                self._config[config_key] = getattr(args, arg_name)
        
        # Update other relevant arguments
        if hasattr(args, 'white_background'):
            self._config['white_background'] = args.white_background
        if hasattr(args, 'debug'):
            self._config['debug'] = args.debug
    
    def save_to_file(self, config_path: str):
        """Save current configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=True)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration value"""
        if name.startswith('_'):
            return super().__getattribute__(name)
        return self._config.get(name, None)
    
    def __setattr__(self, name: str, value: Any):
        """Set configuration value"""
        if name.startswith('_') or name in ['defaults']:
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_config'):
                super().__setattr__(name, value)
            else:
                self._config[name] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self._config.get(key, default)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with dictionary"""
        self._config.update(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self._config.copy()
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return yaml.dump(self._config, default_flow_style=False, sort_keys=True)


def create_default_config(output_path: str):
    """Create a default configuration file"""
    config = Config()
    config.save_to_file(output_path)
    print(f"Default configuration created at {output_path}")


if __name__ == '__main__':
    # Create default config if run as script
    import sys
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = 'config/default.yaml'
    
    create_default_config(output_path)

