## Env
NVIDIA A100-PCIE-40GB

## Base
### Run
```
bash sample/t2v/run.sh
```

### Output
```
Warmup time: 39.553s
Inference time: 38.903s
Iterations per second: 1.569
Max used CUDA memory : 19.933GiB
```


## Compile
Enable `use_compile` in `configs/t2v/t2v_sample.yaml` to enable onediff compile.
### Run
```
bash sample/t2v/run.sh
```

### Output
```
Warmup time: 208.849s
Inference time: 27.059s(-32%)
Iterations per second: 2.225(+42%)
Max used CUDA memory : 18.207GiB
```
