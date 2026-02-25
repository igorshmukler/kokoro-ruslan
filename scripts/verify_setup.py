#!/usr/bin/env python3
"""
Verify MFA installation and dependencies
"""

import sys
import subprocess

def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def check_mfa_installation():
    """Check if MFA command is available"""
    try:
        result = subprocess.run(
            ["mfa", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ Montreal Forced Aligner is installed: {version}")
            return True
        else:
            print("✗ MFA command failed")
            return False
    except FileNotFoundError:
        print("✗ Montreal Forced Aligner is NOT installed")
        return False
    except subprocess.TimeoutExpired:
        print("✗ MFA command timed out")
        return False

def main():
    """Main verification function"""
    print("="*60)
    print("Kokoro-Ruslan MFA Setup Verification")
    print("="*60)
    print()

    all_ok = True

    # Check core dependencies
    print("Checking core dependencies...")
    all_ok &= check_python_package("torch")
    all_ok &= check_python_package("torchaudio")
    all_ok &= check_python_package("numpy")
    print()

    # Check MFA-specific dependencies
    print("Checking MFA dependencies...")
    all_ok &= check_python_package("tgt")
    print()

    # Check MFA installation
    print("Checking MFA installation...")
    mfa_ok = check_mfa_installation()
    all_ok &= mfa_ok
    print()

    # Summary
    print("="*60)
    if all_ok:
        print("✓ All checks passed! You're ready to use MFA.")
        print()
        print("Next steps:")
        print("  1. Run: python preprocess.py --corpus ./ruslan_corpus")
        print("  2. Then: python training.py")
    else:
        print("✗ Some checks failed. Please install missing components.")
        print()
        print("Installation commands:")

        if not mfa_ok:
            print("  # Install MFA (choose one):")
            print("  conda install -c conda-forge montreal-forced-aligner kalpy kaldi")
            print("  # OR")
            print("  pip install montreal-forced-aligner")
            print()

        print("  # Install Python dependencies:")
        print("  pip install -r requirements.txt")
    print("="*60)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
