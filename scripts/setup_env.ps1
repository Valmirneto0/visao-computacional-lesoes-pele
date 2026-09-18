# Setup do ambiente no Windows (rodar no PowerShell, dentro da pasta do projeto).
# Pré-requisito: Python 3.12 instalado (python.org). Python 3.14 não tem wheels de torch/ROCm garantidos.
#   Verifique:  py -3.12 --version
#
# 1) venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2) PyTorch para AMD RX 9060 XT (gfx1200) no Windows — ROCm 7.2.1, PyTorch 2.9.1, Python 3.12
#    Pré-requisito: driver AMD Adrenalin >= 26.2.2
#    Fonte: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
$R = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
pip install --no-cache-dir "$R/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" "$R/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" "$R/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" "$R/rocm-7.2.1.tar.gz"
pip install --no-cache-dir "$R/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "$R/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
#    Fallback CPU (suficiente para a Entrega 1):
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) Demais dependências
pip install -r requirements.txt
python -m ipykernel install --user --name skinlesion --display-name "Python (skinlesion)"

# 4) Teste da GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU')"
