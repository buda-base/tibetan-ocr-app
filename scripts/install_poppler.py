#!/usr/bin/env python3
"""
Script to install Poppler for development on macOS, Windows, and Linux.
This script automates the process used in the GitHub Actions workflow.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def run_command(cmd, shell=False):
    """Run a command and return its output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def ensure_dir_exists(directory):
    """Ensure a directory exists and is writable."""
    try:
        os.makedirs(directory, exist_ok=True)
        # Test if we can write to it
        test_file = os.path.join(directory, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except (PermissionError, OSError) as e:
        print(f"Error: Cannot write to directory {directory}: {e}")
        return False


def safe_copy(src, dst):
    """Safely copy a file, handling permissions."""
    try:
        # Check if destination exists and is writable
        if os.path.exists(dst):
            if not os.access(dst, os.W_OK):
                print(f"Warning: Cannot write to {dst}, trying to change permissions")
                try:
                    os.chmod(dst, 0o644)  # Make writable
                except PermissionError:
                    print(f"Error: Cannot change permissions of {dst}")
                    return False
            # Remove existing file
            os.remove(dst)

        # Copy the file
        shutil.copy2(src, dst)

        # Make executable if it's a binary
        if os.path.dirname(dst).endswith("bin"):
            try:
                os.chmod(dst, 0o755)  # rwxr-xr-x
            except PermissionError:
                print(f"Warning: Could not make {dst} executable")

        return True
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False


def install_poppler_windows():
    """Install Poppler for Windows."""
    print("Installing Poppler for Windows...")

    # Create directory for Poppler
    poppler_dir = Path("poppler")
    poppler_bin = poppler_dir / "bin"

    if not ensure_dir_exists(poppler_dir) or not ensure_dir_exists(poppler_bin):
        print("Error: Cannot create poppler directories")
        return False

    try:
        # Get the latest release information from GitHub API
        with urllib.request.urlopen(
            "https://api.github.com/repos/oschwartz10612/poppler-windows/releases/latest"
        ) as response:
            release_info = json.loads(response.read().decode())

        latest_version = release_info["tag_name"][1:]  # Remove the 'v' prefix
        download_url = next(
            (asset["browser_download_url"] for asset in release_info["assets"] if asset["name"].endswith(".zip")), None
        )

        if not download_url:
            print("Error: Could not find download URL for Poppler")
            return False

        print(f"Latest Poppler version: {latest_version}")
        print(f"Download URL: {download_url}")

        # Download and extract the latest Poppler for Windows
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "poppler.zip")
            print(f"Downloading Poppler to {zip_path}...")
            urllib.request.urlretrieve(download_url, zip_path)

            extract_dir = os.path.join(temp_dir, "poppler_extract")
            os.makedirs(extract_dir, exist_ok=True)

            print(f"Extracting to {extract_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find pdfinfo.exe in the extracted content
            print("Searching for pdfinfo.exe in the extracted content...")
            pdfinfo_path = None
            for root, _, files in os.walk(extract_dir):
                if "pdfinfo.exe" in files:
                    pdfinfo_path = os.path.join(root, "pdfinfo.exe")
                    break

            if not pdfinfo_path:
                print("Error: Could not find pdfinfo.exe in the extracted content")
                return False

            bin_dir = os.path.dirname(pdfinfo_path)
            print(f"Found pdfinfo.exe at: {pdfinfo_path}")
            print(f"Bin directory: {bin_dir}")

            # Copy all files from the bin directory to poppler/bin
            print(f"Copying all files from {bin_dir} to poppler/bin...")
            success = True
            for file in os.listdir(bin_dir):
                src = os.path.join(bin_dir, file)
                dst = os.path.join(poppler_bin, file)
                if not safe_copy(src, dst):
                    success = False

            # Verify that files were copied successfully
            if os.path.exists(os.path.join(poppler_bin, "pdfinfo.exe")) and success:
                print("Poppler binaries copied successfully.")
                print("Final list of files in poppler/bin:")
                for file in os.listdir(poppler_bin):
                    print(f"  - {file}")
            else:
                print("Error: Failed to copy files to poppler/bin")
                return False

        print("Poppler for Windows installed successfully!")
        return True
    except Exception as e:
        print(f"Error installing Poppler for Windows: {e}")
        return False


def install_poppler_macos():
    """Install Poppler for macOS using Homebrew."""
    print("Installing Poppler for macOS...")

    # Create poppler directory structure
    poppler_dir = Path("poppler")
    poppler_bin = poppler_dir / "bin"
    poppler_lib = poppler_dir / "lib"

    if not ensure_dir_exists(poppler_dir) or not ensure_dir_exists(poppler_bin):
        print("Error: Cannot create poppler directories")
        return False

    # Create lib directory
    if not ensure_dir_exists(poppler_lib):
        print("Error: Cannot create poppler/lib directory")
        return False

    try:
        # Check if Homebrew is installed
        try:
            run_command(["brew", "--version"])
        except:
            print("Homebrew is not installed. Please install it first: https://brew.sh")
            return False

        # Install Poppler using Homebrew
        run_command(["brew", "install", "poppler"])
        brew_prefix = run_command(["brew", "--prefix"])
        poppler_prefix = run_command(["brew", "--prefix", "poppler"])
        print(f"Homebrew prefix: {brew_prefix}")
        print(f"Poppler installation directory: {poppler_prefix}")

        # Get the Homebrew lib directory
        brew_lib = os.path.join(brew_prefix, "lib")
        print(f"Homebrew lib directory: {brew_lib}")

        # Copy binaries from Homebrew
        binaries = ["pdfinfo", "pdftoppm", "pdftotext"]
        success = True
        for binary in binaries:
            src = os.path.join(poppler_prefix, "bin", binary)
            dst = os.path.join(poppler_bin, binary)
            if os.path.exists(src):
                if not safe_copy(src, dst):
                    success = False
                    print(f"Warning: Could not copy {binary}")
                else:
                    print(f"Copied {binary} to {dst}")
            else:
                print(f"Warning: Could not find {binary} at {src}")
                success = False

        # Copy Poppler libraries
        print("Copying Poppler libraries...")

        # Find all libpoppler*.dylib files in the Homebrew lib directory
        lib_files = []
        if os.path.exists(brew_lib):
            for file in os.listdir(brew_lib):
                if file.startswith("libpoppler") and file.endswith(".dylib"):
                    lib_files.append(file)

        # If no libraries found, try to find them in other common locations
        if not lib_files:
            print("No Poppler libraries found in primary location, checking alternatives...")
            alt_lib_dirs = [os.path.join(brew_prefix, "opt", "poppler", "lib"), "/usr/local/lib", "/opt/homebrew/lib"]

            for alt_dir in alt_lib_dirs:
                if os.path.exists(alt_dir):
                    print(f"Checking {alt_dir}...")
                    for file in os.listdir(alt_dir):
                        if file.startswith("libpoppler") and file.endswith(".dylib"):
                            lib_files.append(file)
                            brew_lib = alt_dir  # Update the lib directory
                            break

                    if lib_files:
                        print(f"Found Poppler libraries in {alt_dir}")
                        break

        if not lib_files:
            print("Warning: Could not find any Poppler libraries")
            success = False
        else:
            # Copy the library files
            for lib_file in lib_files:
                src = os.path.join(brew_lib, lib_file)
                dst = os.path.join(poppler_lib, lib_file)
                if not safe_copy(src, dst):
                    print(f"Warning: Could not copy {lib_file}")
                    success = False
                else:
                    print(f"Copied {lib_file} to {dst}")

            # Find the actual libpoppler version
            actual_libpoppler = None
            for lib_file in lib_files:
                if lib_file.startswith("libpoppler.") and lib_file.count(".") >= 2:
                    actual_libpoppler = lib_file
                    break

            if actual_libpoppler:
                # Create symlinks for all possible versions to ensure compatibility
                version_parts = actual_libpoppler.split(".")
                if len(version_parts) >= 3 and version_parts[1].isdigit():
                    base_version = int(version_parts[1])
                    # Create symlinks for a range of versions around the current one
                    for v in range(base_version - 5, base_version + 5):
                        symlink_name = f"libpoppler.{v}.dylib"
                        symlink_path = os.path.join(poppler_lib, symlink_name)
                        if not os.path.exists(symlink_path) and v != base_version:
                            try:
                                os.symlink(actual_libpoppler, symlink_path)
                                print(f"Created symlink from {actual_libpoppler} to {symlink_name}")
                            except Exception as e:
                                print(f"Warning: Could not create symlink for {symlink_name}: {e}")

        # Create a README file with instructions
        readme_path = os.path.join(poppler_dir, "README.txt")
        try:
            with open(readme_path, "w") as f:
                f.write(
                    """POPPLER INSTALLATION FOR MACOS

The Poppler binaries have been copied to the 'bin' directory and libraries to the 'lib' directory.

For the application to work properly with PDFs, you need to have Poppler installed via Homebrew:
    brew install poppler

If you experience any issues with PDF processing, please ensure that Poppler is correctly
installed via Homebrew and that the binaries in the 'bin' directory are executable.

Library compatibility:
- The script has created symlinks for different libpoppler versions to ensure compatibility
- If you still encounter library loading issues, you may need to manually create additional symlinks
  or reinstall Poppler with Homebrew
"""
                )
            print(f"Created README file at {readme_path}")
        except Exception as e:
            print(f"Warning: Could not create README file: {e}")
            success = False

        if success:
            print("Poppler for macOS installed successfully!")
            print("Libraries have been copied and symlinks created for compatibility.")
            return True
        else:
            print("Warning: Some steps failed during installation.")
            print("The application may still work if Poppler is installed via Homebrew.")
            return True  # Return True anyway since the app can use system Poppler
    except Exception as e:
        print(f"Error installing Poppler for macOS: {e}")
        return False


def install_poppler_linux():
    """Install Poppler for Linux."""
    print("Installing Poppler for Linux...")

    # Create poppler directory structure
    poppler_dir = Path("poppler")
    poppler_bin = poppler_dir / "bin"

    if not ensure_dir_exists(poppler_dir) or not ensure_dir_exists(poppler_bin):
        print("Error: Cannot create poppler directories")
        return False

    try:
        # Check the Linux distribution
        if os.path.exists("/etc/debian_version"):
            # Debian/Ubuntu
            print("Detected Debian/Ubuntu system")
            run_command(["sudo", "apt-get", "update"])
            run_command(["sudo", "apt-get", "install", "-y", "poppler-utils", "libpoppler-dev"])
        elif os.path.exists("/etc/fedora-release"):
            # Fedora
            print("Detected Fedora system")
            run_command(["sudo", "dnf", "install", "-y", "poppler-utils", "poppler-devel"])
        elif os.path.exists("/etc/arch-release"):
            # Arch Linux
            print("Detected Arch Linux system")
            run_command(["sudo", "pacman", "-Sy", "poppler"])
        else:
            print("Unsupported Linux distribution. Please install Poppler manually.")
            return False

        # Copy binaries to our directory
        binaries = ["pdfinfo", "pdftoppm", "pdftotext"]
        success = True
        for binary in binaries:
            src = shutil.which(binary)
            if src:
                dst = os.path.join(poppler_bin, os.path.basename(binary))
                try:
                    if not safe_copy(src, dst):
                        print(f"Creating a symbolic link instead...")
                        if os.path.exists(dst):
                            os.remove(dst)
                        os.symlink(src, dst)
                        print(f"Created symlink from {src} to {dst}")
                except Exception as e:
                    print(f"Warning: Could not copy or link {binary}: {e}")
                    success = False
            else:
                print(f"Warning: Could not find {binary} in PATH")
                success = False

        # Create a README file with instructions
        readme_path = os.path.join(poppler_dir, "README.txt")
        try:
            with open(readme_path, "w") as f:
                f.write(
                    """POPPLER INSTALLATION FOR LINUX

The Poppler binaries have been copied or linked to the 'bin' directory.

For the application to work properly with PDFs, you need to have Poppler installed:
    Debian/Ubuntu: sudo apt-get install poppler-utils libpoppler-dev
    Fedora: sudo dnf install poppler-utils poppler-devel
    Arch Linux: sudo pacman -S poppler

The application will look for Poppler in this directory structure, but will fall back to using
the system-installed version if needed.

If you experience any issues with PDF processing, please ensure that Poppler is correctly
installed and that the binaries in the 'bin' directory are executable.
"""
                )
            print(f"Created README file at {readme_path}")
        except Exception as e:
            print(f"Warning: Could not create README file: {e}")
            success = False

        if success:
            print("Poppler for Linux installed successfully!")
        else:
            print("Warning: Some steps failed during installation.")
            print("The application may still work if Poppler is installed system-wide.")

        return True  # Return True anyway since the app can use system Poppler
    except Exception as e:
        print(f"Error installing Poppler for Linux: {e}")
        return False


def main():
    """Main function to install Poppler based on the platform."""
    system = platform.system()

    if system == "Windows":
        success = install_poppler_windows()
    elif system == "Darwin":  # macOS
        success = install_poppler_macos()
    elif system == "Linux":
        success = install_poppler_linux()
    else:
        print(f"Unsupported platform: {system}")
        sys.exit(1)

    if success:
        print("\nPoppler installation completed successfully!")
        print("You can now run the application with 'python main.py'")
    else:
        print("\nPoppler installation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
