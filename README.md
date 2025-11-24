## Benchmark Device-initiated NCCL

### Description
Benchmark for [Device-initiated NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)(introduced in NCCL 2.28.7)

### Getting Started
1. Install CUDA with NCCL(>=2.28.7)
2. Install [uv](https://docs.astral.sh/uv/)
3. Run following commands to setup project:
```bash
# Setup uv venv
uv sync

# Install pre-commit
pre-commit install
```
4. Build the binaries
```bash
uv run --no-sync cmake -B build -S . -G Ninja
uv run --no-sync cmake --build build
```

#### Example: Add One
```bash
# Run add one example
./build/run_add_one
```

#### Example: LSA All-Reduce
```bash
# Run LSA All-Reduce example
uv run --no-sync mpirun -n 2 ./build/run_lsa_all_reduce
```
