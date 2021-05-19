#!/usr/bin/env bash
set -e

function log {
    local PURPLE='\033[0;35m'
    local NOCOLOR='\033[m'
    local BOLD='\033[1m'
    local NOBOLD='\033[0m'
    echo -e -n "${PURPLE}${BOLD}$1${NOBOLD}${NOCOLOR}" >&2
}

function install_vscode_extensions {
    export PATH="$PATH:/Applications/Visual Studio Code.app/Contents/Resources/app/bin"
    if [ -x "$(command -v code)" ]
    then
        log "Installing VS Code's Python extension...\\n"
        code --install-extension ms-python.python --force
        code --install-extension ms-python.vscode-pylance --force
        log "Done!\\n"
    else
        log "VS Code CLI not found, skipping installation of extensions..."
    fi
}

function install_miniconda {
    if ! [ -x "$(command -v conda)" ]
    then
        log "Installing conda... "
        if [ "$(uname)" == "Darwin" ]; then local PLATFORM=MacOSX; else local PLATFORM=Linux; fi
        curl --silent https://repo.anaconda.com/miniconda/Miniconda3-latest-"$PLATFORM"-x86_64.sh --output /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -f > /dev/null
        export PATH="$HOME/miniconda3/bin:$PATH"
        conda init > /dev/null
        log "Done!\\n"
    fi
    rm /tmp/miniconda.sh 2> /dev/null || true
}

function update_conda {
    log "Updating conda... "
    conda install mamba --channel conda-forge --yes > /dev/null
    mamba update --name base --yes conda > /dev/null
    log "Done!\\n"
}

function create_conda_env {
    log "Removing deprecated .envs directory... "
    local SCRIPT_PATH
    SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
    rm -rf "$SCRIPT_PATH"/../.envs/
    conda config --remove envs_dirs .envs 2> /dev/null || true
    log "Done!\\n"
    log "Creating conda environment agoro-field-boundary-detector-env...\\n\\n"
    pip install --quiet conda-merge
    conda-merge environment.run.yml environment.dev.yml > environment.yml
    mamba env create --force
    rm environment.yml
    log "Done!\\n"
    log "Installing editable agoro_field_boundary_detector into conda environment...\\n\\n"
    # shellcheck disable=SC1091
    source activate agoro-field-boundary-detector-env
    pip install --editable .
    conda deactivate
    log "Done!\\n"
}

function configure_git {
    log "Installing pre-commit hooks...\\n"
    # shellcheck disable=SC1091
    source activate agoro-field-boundary-detector-env
    pre-commit install --hook-type pre-commit
    pre-commit install --hook-type prepare-commit-msg
    pre-commit install --hook-type pre-push
    pre-commit run --all-files || true
    log "Done!\\n"
    log "Enabling git push.followTags... "
    git config --local push.followTags true
    conda deactivate
    log "Done!\\n"
}

function install_jupyter_widgets {
    log "Installing Jupyter widgets...\\n"
    # shellcheck disable=SC1091
    source activate agoro-field-boundary-detector-env
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    conda deactivate
    log "Done!\\n"
}

function install_fish {
    log "Installing Fish and Starship...\\n"
    apt-get update && apt-get -y install fish curl
    chsh --shell /usr/bin/fish
    curl -fsSL https://starship.rs/install.sh > install.sh && chmod +x install.sh && ./install.sh --yes && rm install.sh
    conda init fish
    conda config --set changeps1 False
    # shellcheck disable=SC2129
    echo "set fish_greeting" >> ~/.config/fish/config.fish
    echo "starship init fish | source" >> ~/.config/fish/config.fish
    echo "conda activate agoro-field-boundary-detector-env" >> ~/.config/fish/config.fish
    log "Done!\\n"
}

function list_tasks {
    log "Local environment ready! Remember to activate your conda environment with:\\n\\n"
    log "$ conda activate agoro-field-boundary-detector-env\\n\\n"
    log "After which you can list the available tasks with:\\n\\n"
    log "$ invoke --list\\n\\n"
    # shellcheck disable=SC1091
    source activate agoro-field-boundary-detector-env
    invoke --list
    conda deactivate
}

function run_command {
    local COMMAND=$1
    case $COMMAND in
    help|--help)
        cat << EOF
Usage: ./init.sh

Running this script will:

  1. Install VS Code's Python extension.
  2. Install conda for the current user, if not already installed.
  3. Update conda to the latest version.
  4. Create the conda environment specified by the union of environment.run.yml and environment.dev.yml.
  5. Install pre-commit hooks and configure git.
EOF
        ;;
    devcontainer|--devcontainer)
        install_fish
        update_conda
        create_conda_env
        configure_git
        # install_jupyter_widgets
        ;;
    *)
        install_vscode_extensions
        install_miniconda
        update_conda
        create_conda_env
        configure_git
        # install_jupyter_widgets
        list_tasks
        ;;
    esac
}

run_command "$@"
