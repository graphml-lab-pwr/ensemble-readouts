all: quality
check_dirs := readouts experiments tests
# Check that source code meets quality standards

quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	mypy --install-types --non-interactive $(check_dirs)

fix:
	black $(check_dirs)
	isort $(check_dirs)

install_cpu:
	pip install -r requirements.txt -r requirements-cpu.txt

install_gpu:
	pip install -r requirements.txt -r requirements-gpu.txt
