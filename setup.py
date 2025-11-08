#!/usr/bin/env python3
"""
Traffic Anomaly Detection System - Setup Script
Automates the initial setup and verification process
"""

import os
import sys
from pathlib import Path
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    base_dir = Path(__file__).parent
    directories = [
        base_dir / "models",
        base_dir / "temp",
        base_dir / "outputs"
    ]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True)
            print(f"✅ Created: {directory}")
        else:
            print(f"ℹ️  Already exists: {directory}")
    
    return True

def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        return False
    
    print("Installing packages... This may take a few minutes.")
    print("-" * 60)
    
    try:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_file),
            "--upgrade"
        ])
        print("-" * 60)
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_model_file():
    """Check if model file exists"""
    print_header("Checking Model File")
    
    model_path = Path(__file__).parent / "models" / "best_model.h5"
    
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {model_path}")
        print(f"   File size: {file_size:.2f} MB")
        return True
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("\n   Please place your trained model file 'best_model.h5' in the models/ directory")
        print("   The application will not work without this file.")
        return False

def verify_imports():
    """Verify critical imports"""
    print_header("Verifying Package Imports")
    
    packages = [
        ("tensorflow", "TensorFlow"),
        ("cv2", "OpenCV"),
        ("streamlit", "Streamlit"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("plotly", "Plotly"),
        ("pywt", "PyWavelets")
    ]
    
    all_ok = True
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} import successful")
        except ImportError:
            print(f"❌ {name} import failed")
            all_ok = False
    
    return all_ok

def create_run_script():
    """Create a simple run script"""
    print_header("Creating Run Script")
    
    base_dir = Path(__file__).parent
    
    # Windows batch file
    batch_file = base_dir / "run.bat"
    batch_content = """@echo off
echo Starting Traffic Anomaly Detection System...
streamlit run app.py
pause
"""
    
    with open(batch_file, 'w') as f:
        f.write(batch_content)
    print(f"✅ Created: {batch_file}")
    
    # Unix shell script
    shell_file = base_dir / "run.sh"
    shell_content = """#!/bin/bash
echo "Starting Traffic Anomaly Detection System..."
streamlit run app.py
"""
    
    with open(shell_file, 'w') as f:
        f.write(shell_content)
    
    # Make executable on Unix
    try:
        os.chmod(shell_file, 0o755)
        print(f"✅ Created: {shell_file}")
    except:
        print(f"ℹ️  Created: {shell_file} (may need chmod +x)")
    
    return True

def print_summary(results):
    """Print setup summary"""
    print_header("Setup Summary")
    
    all_ok = all(results.values())
    
    for step, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {step}")
    
    print("\n")
    
    if all_ok:
        print("🎉 Setup completed successfully!")
        print("\nTo start the application:")
        print("  • Run: streamlit run app.py")
        print("  • Or use the run script: ./run.sh (Unix) or run.bat (Windows)")
    else:
        print("⚠️  Setup completed with warnings/errors")
        print("\nPlease address the issues above before running the application")
    
    print("\n" + "=" * 60 + "\n")

def main():
    """Main setup function"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   Traffic Anomaly Detection System - Setup Wizard        ║
║                     Version 1.0                          ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Run setup steps
    results["Python Version Check"] = check_python_version()
    
    if not results["Python Version Check"]:
        print("\n❌ Setup cannot continue with incompatible Python version")
        sys.exit(1)
    
    results["Directory Creation"] = create_directories()
    results["Dependency Installation"] = install_dependencies()
    results["Package Import Verification"] = verify_imports()
    results["Model File Check"] = check_model_file()
    results["Run Script Creation"] = create_run_script()
    
    # Print summary
    print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()