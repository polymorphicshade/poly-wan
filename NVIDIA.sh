#!/usr/bin/env bash
# Sets up an Ubuntu or Debian server so the poly-wan container can use the NVIDIA GPU:
# the NVIDIA driver (570 or newer), Docker, and the NVIDIA Container Toolkit.
#
#   sudo ./NVIDIA.sh
#
# Safe to run again: each step is skipped when it is already done. If a new driver is installed
# the script asks you to reboot; run it again afterwards to finish and verify the setup.
#
# On Ubuntu the driver uses Ubuntu's prebuilt, signed kernel modules rather than DKMS, which
# compiles the module on the server and breaks when a kernel update lands. An existing DKMS
# driver is removed and replaced.
#
# Environment overrides:
#   NVIDIA_DRIVER_BRANCH    Ubuntu: driver branch instead of the newest one found (e.g. 570-server-open)
#   NVIDIA_DRIVER_PACKAGE   Debian: driver package instead of nvidia-open
#   SKIP_DOCKER_INSTALL=1   do not install Docker when it is missing
set -euo pipefail

MIN_DRIVER=570
TEST_IMAGE=ubuntu:24.04

log() { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------- checks
[ "$(uname -s)" = "Linux" ] || die "This script is for the Linux server, not this machine."
[ "$(id -u)" -eq 0 ] || die "Run it as root: sudo $0"

[ -r /etc/os-release ] || die "Cannot read /etc/os-release to identify the distribution."
. /etc/os-release
case "${ID:-}" in
    ubuntu | debian) ;;
    *) die "Unsupported distribution '${ID:-unknown}'. This script supports Ubuntu and Debian." ;;
esac

export DEBIAN_FRONTEND=noninteractive

# Installed packages of an NVIDIA driver, leaving out the container toolkit's packages.
installed_driver_packages() {
    dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' 2>/dev/null \
        | awk '$1 ~ /^.[iUFHWt]/ {print $2}' \
        | grep -E '^(nvidia-|libnvidia-|linux-modules-nvidia-|linux-objects-nvidia-|linux-signatures-nvidia-|xserver-xorg-video-nvidia)' \
        | grep -vE '^(nvidia-container-|libnvidia-container)' || true
}

# A DKMS driver that fails to build for a new kernel leaves dpkg half-configured, and every apt
# command fails until it is gone.
if [ "$ID" = "ubuntu" ] && installed_driver_packages | grep -q '^nvidia-dkms-'; then
    log "Removing the DKMS-built NVIDIA driver; a prebuilt one replaces it"
    for module in $(dkms status 2>/dev/null | grep -oE '^nvidia/[^,: ]+' | sort -u); do
        dkms remove "$module" --all || true
    done
    mapfile -t old_packages < <(installed_driver_packages)
    apt-get purge -y -q "${old_packages[@]}" || dpkg --purge --force-depends "${old_packages[@]}"
fi
dpkg --configure -a || die "dpkg could not finish configuring packages; see the errors above."
apt-get install -f -y -q

apt-get update -q
apt-get install -y -q --no-install-recommends ca-certificates curl gnupg pciutils

lspci | grep -qi 'nvidia' || die "No NVIDIA GPU found (lspci shows none)."

# Prints the loaded driver's major version, or nothing when no driver is loaded.
driver_major() {
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1
}

# ---------------------------------------------------------------------------- driver
install_driver_ubuntu() {
    # Kernel flavour (generic, lowlatency, aws, ...) and the meta package the modules follow:
    # the HWE one when the HWE kernel is installed, so kernel updates bring matching modules.
    local flavour meta
    flavour=$(uname -r | cut -d- -f3-)
    meta=$flavour
    if dpkg-query -W -f='${db:Status-Abbrev}' "linux-image-${flavour}-hwe-${VERSION_ID}" 2>/dev/null | grep -q '^ii'; then
        meta="${flavour}-hwe-${VERSION_ID}"
    fi

    local branch="${NVIDIA_DRIVER_BRANCH:-}"
    if [ -z "$branch" ]; then
        # Newest open-kernel-module branch with prebuilt modules, server branches first.
        branch=$(apt-cache pkgnames linux-modules-nvidia- \
            | sed -nE "s/^linux-modules-nvidia-([0-9]+-server-open)-${meta}\$/\1/p" | sort -V | tail -n1)
        [ -n "$branch" ] || branch=$(apt-cache pkgnames linux-modules-nvidia- \
            | sed -nE "s/^linux-modules-nvidia-([0-9]+-open)-${meta}\$/\1/p" | sort -V | tail -n1)
    fi
    [ -n "$branch" ] || die "No prebuilt NVIDIA modules found for the '$meta' kernel. Set NVIDIA_DRIVER_BRANCH."
    [ "${branch%%-*}" -ge "$MIN_DRIVER" ] \
        || die "Newest driver available is $branch; $MIN_DRIVER or newer is needed. Upgrade Ubuntu or set NVIDIA_DRIVER_BRANCH."

    # nvidia-smi ships in nvidia-utils, which is not split into open and closed packages.
    local utils="nvidia-utils-${branch}"
    apt-cache show "$utils" >/dev/null 2>&1 || utils="nvidia-utils-${branch%-open}"

    local pkgs=("linux-modules-nvidia-${branch}-${meta}" "nvidia-headless-no-dkms-${branch}" "$utils")
    for pkg in "${pkgs[@]}"; do
        apt-cache show "$pkg" >/dev/null 2>&1 || die "Package $pkg is not available. Set NVIDIA_DRIVER_BRANCH to another branch."
    done

    log "Installing NVIDIA driver $branch (prebuilt modules for the '$meta' kernel)"
    apt-get install -y -q "${pkgs[@]}"
}

install_driver_debian() {
    # Debian's own archive lags behind; use NVIDIA's CUDA repository for a current driver.
    local arch repo
    case "$(uname -m)" in
        x86_64) arch=x86_64 ;;
        aarch64) arch=sbsa ;;
        *) die "Unsupported CPU architecture $(uname -m)." ;;
    esac
    repo="https://developer.download.nvidia.com/compute/cuda/repos/debian${VERSION_ID%%.*}/${arch}"

    if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
        log "Adding NVIDIA's CUDA repository ($repo)"
        local deb
        deb=$(mktemp --suffix=.deb)
        curl -fsSL "$repo/cuda-keyring_1.1-1_all.deb" -o "$deb" \
            || die "Could not download the CUDA keyring for Debian ${VERSION_ID}. Is this release supported by NVIDIA?"
        dpkg -i "$deb"
        rm -f "$deb"
        apt-get update -q
    fi

    local pkg="${NVIDIA_DRIVER_PACKAGE:-nvidia-open}"
    log "Installing $pkg"
    apt-get install -y -q "linux-headers-$(uname -r)" "$pkg"
}

needs_reboot=0
current=$(driver_major || true)
if [ -n "$current" ] && [ "$current" -ge "$MIN_DRIVER" ]; then
    log "NVIDIA driver $current is already loaded"
else
    if [ -n "$current" ]; then
        warn "Loaded driver $current is older than $MIN_DRIVER; upgrading."
    fi
    if command -v mokutil >/dev/null 2>&1 && mokutil --sb-state 2>/dev/null | grep -qi 'enabled'; then
        warn "Secure Boot is on. The driver's kernel module must be signed: if prompted, set a MOK password" \
            "and enroll the key at the blue MOK screen on the next boot, or disable Secure Boot."
    fi
    case "$ID" in
        ubuntu) install_driver_ubuntu ;;
        debian) install_driver_debian ;;
    esac
    needs_reboot=1
fi

# ---------------------------------------------------------------------------- docker
if command -v docker >/dev/null 2>&1; then
    log "Docker is already installed ($(docker --version))"
elif [ "${SKIP_DOCKER_INSTALL:-0}" = "1" ]; then
    die "Docker is not installed and SKIP_DOCKER_INSTALL=1 was set."
else
    log "Installing Docker Engine and the compose plugin"
    curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ] && ! id -nG "$SUDO_USER" | grep -qw docker; then
    log "Adding $SUDO_USER to the docker group (takes effect at next login)"
    usermod -aG docker "$SUDO_USER"
fi

# ---------------------------------------------------------------------------- container toolkit
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    log "Installing the NVIDIA Container Toolkit"
    keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o "$keyring"
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -q
    apt-get install -y -q nvidia-container-toolkit
else
    log "NVIDIA Container Toolkit is already installed"
fi

log "Registering the NVIDIA runtime with Docker"
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# ---------------------------------------------------------------------------- verify
if [ "$needs_reboot" -eq 1 ] && [ -z "$(driver_major || true)" ]; then
    log "The driver is installed but not loaded yet."
    echo "Reboot the server (sudo reboot), then run this script again to verify the GPU works in Docker."
    exit 0
fi

log "Checking GPU access from a container"
if docker run --rm --gpus all "$TEST_IMAGE" nvidia-smi; then
    log "Done. The GPU is available to Docker. Start the app with: docker compose up -d --build"
else
    die "The test container could not reach the GPU. Check 'nvidia-smi' on the host and /etc/docker/daemon.json."
fi
