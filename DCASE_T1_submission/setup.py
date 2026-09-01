from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="DACES_T1",
    version="0.1.0",
    description="DCASE 2025 Task 1 V4 LC-AFF inference package",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "DACES_T1": [
            "resources/*.wav",
            "resources/*.csv",
            "resources/*.txt",
            "ckpts/*.ckpt",
            "ckpts/*.pt",
            "ckpts/*.txt",
        ]
    },
    python_requires=">=3.10",
)
