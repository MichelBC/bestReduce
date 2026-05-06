# BestReduce: Optimized Parallel Reduction for Regular and Irregular Segments on GPU

---

# Citation

Please cite the corresponding papers if it was useful for your research:

```bibtex
@article{cordeiro2025bestreduce,
  title={Optimized Parallel Reduction for Regular and Irregular Segments on GPU},
  author={Cordeiro, Michel B. and Zola, Wagner M. N.},
  journal={Concurrency and Computation: Practice and Experience},
  year={2025}
}
```

---

# Overview

**BestReduce** is a high-performance CUDA implementation of:

* Non-segmented parallel reduction
* Segmented parallel reduction (regular and irregular segments)

The core contribution is a **runtime-adaptive segmented reduction algorithm** that dynamically selects the most efficient kernel based on the segment size.

Unlike existing GPU libraries that rely on a fixed granularity model, BestReduce implements multiple specialized kernels and selects among them automatically. Additionally, it includes a tuning algorithm that identifies the best kernel for different segment size ranges.

---

## Compilation

First, edit the `Makefile` if necessary:

```make
CUDA_ARCH := sm_86   # change if needed
```

Examples:

* RTX A4500 → `sm_86`
* RTX 4080 → `sm_89`

Then, run the following commands to perform the tuning and build the optimized version of `bestReduce`:

```bash
make tuning-bestReduce
./tuning-bestReduce
make bestReduce bestReduce64
```

Although tuning is **not mandatory**, it is essential for achieving the best performance on your machine.

---

## Running

To run the program, use the following command:

```bash
Usage: TEST="function" <executable> <nElements> [<nSegments>] [-d <dist_name>]
```

Where:

* `function` for non-segmented reduction can be one of the following:

  * `ui32_atomic_op`, `ui32_shfl`, `ui32x4`, `ui32_sh_mem`, `cub`, `thrust`

* `function` for segmented reduction can be one of the following:

  * `segCub`, `segThrust`, `bestReduce`

* Alternatively, you can test an individual kernel for segmented reduction:

  * `segThread`, `segWarp`, `segKernel`, `manyBlocks`

### Notes:

* The number of segments is **optional** for non-segmented reduction, but **required** for segmented reduction.
* The `-d` flag specifies the distribution type, which can be:

  * `regular`, `normal`, or `exp`

If you choose `function="manyBlocks"`, you can set the number of blocks per segment using the `-grain` argument.

### Example:

```bash
TEST="bestReduce" ./bestReduce 1000000 500 -d normal
TEST="bestReduce" ./bestReduce64 4000000 100 -d exp
```

---

## Auto-Tuning Thresholds

The performance of `bestReduce` depends on threshold values that determine which kernel is selected based on the average segment size. However, a default value is provided that works well in most cases, so tuning is **not mandatory**.

To generate optimal thresholds for your GPU, run the following command:

```bash
./tuning-bestReduce
```

This procedure will:

1. Benchmark each kernel
2. Identify intersection points between kernels
3. Generate a `tuning.data` file

After tuning, you can build the optimized versions of `bestReduce` by running:

```bash
make bestReduce bestReduce64
```

### Customizing the Tuning Process

You can customize the number of elements used during tuning by passing the desired number as an argument:

```bash
./tuning-bestReduce [number of elements]
```

This allows you to tune for different data set sizes, enabling the main algorithm to select the best thresholds based on the data set that most closely matches the one being processed during execution.

---

## Reproducibility of Paper Results

The implementation includes comparisons with:

* **NVIDIA CUB library**
* **NVIDIA Thrust library**

All experiments in the paper were performed with:

* 32 million elements
* Maximum reduction operator
* Average throughput reported in **Giga-elements/s**

All result data is available in the LibreOffice sheet located in the **'results'** folder. This file contains detailed instructions on how to generate each table and plot from the paper. The necessary scripts are in the **'scripts'** folder.

### Summary:

#### Non-segmented Experiments:

To reproduce **Table 1** and **Table 2**, run:

```bash
./scripts/do-nonseg-tests-30-times.sh all
```

#### Segmented Experiments:

To reproduce **Figure 1** and **Figure 2**, run:

```bash
./scripts/do-seg-tests-30-times.sh allkernels regular
```

**Note**: Some kernels may take a long time to execute if they are run outside of the range they were designed for.

For **Figure 3**, **Figure 4**, and **Figure 5** (32-bit only), run the following commands, respectively:

```bash
./scripts/do-seg-tests-30-times.sh all regular
./scripts/do-seg-tests-30-times.sh all normal
./scripts/do-seg-tests-30-times.sh all exp
```

For **Figure 6** (both 32-bit and 64-bit), run:

```bash
./scripts/do-seg-tests-30-times-64bits.sh all regular
./scripts/do-seg-tests-30-times-64bits.sh all normal
./scripts/do-seg-tests-30-times-64bits.sh all exp
```

---

## License

This project is licensed under the BSD 3-Clause License.
See the `LICENSE` file for details.

---

## Contact

Michel B. Cordeiro
Federal University of Paraná (UFPR)
Email: [michel.brasil.c@gmail.com](mailto:michel.brasil.c@gmail.com)
