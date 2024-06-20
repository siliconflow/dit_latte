## Env
NVIDIA A100-PCIE-40GB

## Base
### Run
```
bash sample/t2v/run.sh
```

### Output
```
Warmup time: 33.291s
Inference time: 32.618s
Iterations per second: 1.60
Max used CUDA memory : 28.208GiB
```


## Compile
Enable `use_compile` in `configs/t2v/t2v_sample.yaml` to enable onediff compile.
### Run
```
bash sample/t2v/run.sh
```

### Output
```
Warmup time: 572.877s
Inference time: 22.601s(-30.7%)
Iterations per second: 2.27(+41.9%)
Max used CUDA memory : 24.753GiB
```
