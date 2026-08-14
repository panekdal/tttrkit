#!/usr/bin/env python3
"""Test harmonic scanning phase mapping."""

import numpy as np
import sys

# Add path and remove any cached modules
sys.path.insert(0, 'src')
for mod in list(sys.modules.keys()):
    if 'tttrkit' in mod:
        del sys.modules[mod]

from tttrkit.ptuio.reconstructor import ScanConfig, ImageReconstructor


def test_harmonic_phase_mapping():
    """Test harmonic phase to pixel mapping with different phase offsets."""
    print("Testing harmonic phase to pixel mapping...\n")
    print("The scanner oscillates as: position = sin(2πt + φ)")
    print("Different phase offsets shift where in the cycle each line begins.\n")
    
    # Sample over smaller range to show single-direction motion
    test_phases = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Test different phase offsets
    for phase_offset in [0.0, np.pi/6, np.pi/4, np.pi/3]:
        config = ScanConfig(
            lines=256,
            pixels=256,
            frames=1,
            harmonic_scan=True,
            harmonic_duty=phase_offset
        )
        reconstructor = ImageReconstructor(config)
        corrected = reconstructor._harmonic_to_linear_pixel(test_phases)
        
        print(f"Phase offset φ = {phase_offset:.4f} rad ({np.degrees(phase_offset):5.1f}°)")
        print(f"  Temporal:  {test_phases}")
        print(f"  Spatial:   {corrected}")
        print(f"  Range:     [{np.min(corrected):.3f}, {np.max(corrected):.3f}]")
        print()


def test_config_serialization():
    """Test ScanConfig to_dict/from_dict with harmonic parameters."""
    print("Testing ScanConfig serialization with harmonic parameters...")
    
    config = ScanConfig(
        lines=512,
        pixels=512,
        frames=2,
        harmonic_scan=True,
        harmonic_duty=np.pi/4
    )
    
    config_dict = config.to_dict()
    print("Config dict keys:", list(config_dict.keys()))
    
    # Verify harmonic parameters are in dict
    assert "harmonic_scan" in config_dict, "harmonic_scan missing from dict"
    assert "harmonic_phase" in config_dict, "harmonic_phase missing from dict"
    assert config_dict["harmonic_scan"] == True
    assert config_dict["harmonic_phase"] == np.pi/4
    print("✓ Harmonic parameters correctly serialized")
    
    # Test deserialization
    config2 = ScanConfig.from_dict(config_dict)
    assert config2.harmonic_scan == config.harmonic_scan
    assert config2.harmonic_duty == config.harmonic_duty
    print("✓ Harmonic parameters correctly deserialized")


if __name__ == "__main__":
    test_harmonic_phase_mapping()
    test_config_serialization()
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
