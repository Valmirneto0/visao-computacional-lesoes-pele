# Setup do ambiente no Windows (rodar no PowerShell, dentro da pasta do projeto).
# Pré-requisito: Python 3.12 instalado (python.org). Python 3.14 não tem wheels de torch/ROCm garantidos.
#   Verifique:  py -3.12 --version
#
# 1) venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2) PyTorch para AMD RX 9060 XT (RDNA4, gfx1200) no Windows
#    Siga a página oficial — o índice/URL muda por versão do ROCm:
#    https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
#    Exemplo (confira antes de rodar):
#    pip install torch torchvision --index-url https://repo.radeon.com/rocm/windows/<versao>/
#
#    Fallback CPU (suficiente para a Entrega 1):
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) Demais dependências
pip install -r requirements.txt
python -m ipykernel install --user --name skinlesion --display-name "Python (skinlesion)"

# 4) Teste da GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU')"
