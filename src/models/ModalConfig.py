import os
import modal
import dotenv


dotenv.load_dotenv(".env")


class ModalConfig:
    stub = modal.Stub(name=os.getenv("MODAL_STUB_NAME"))
    image = modal.Image.debian_slim().pip_install_from_requirements(
        requirements_txt="./modal_requirements.txt")
    device = modal.gpu.A100(memory=40)
