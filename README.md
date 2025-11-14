# MsktHMM : HMM with multivariate skew-t emissions

`MsktHMM` adds **uMST (unrestricted multivariate skew-t)** emissions on top of
[`hmmlearn`](https://github.com/hmmlearn/hmmlearn):

- emission distribution in each state: **multivariate skew-t (Sahu–Dey–Branco, 2003)**  
- plugged into the standard `hmmlearn` forward–backward / Viterbi / EM framework  
- works on **Windows** (prebuilt EMMIXskew DLLs included, or build locally)  
- works on **Linux** (build EMMIXskew `.so` locally)

---

## 1) Prerequisites

- Python **3.10–3.13**
- A C/Fortran toolchain is needed **only** if you want to (re)build the native
  EMMIXskew library yourself. For normal Windows users the prebuilt DLLs should be enough.

Create a virtual env and install the dependencies (example: Windows PowerShell):

```powershell
powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip setuptools wheel
pip install -e .[dev]
````



---


## 2) Native EMMIXskew backend

MsktHMM relies on the **EMMIXskew** Fortran/C library to:

* evaluate multivariate skew-t densities,
* compute the truncated-t expectations needed in the E-step,
* implement the closed-form M-step of the Sahu–Dey–Branco uMST model.

At runtime, `mskt_hmm.native` tries to load:

* **Windows:** `libemmixskew.dll`
* **Linux:** `libemmixskew.so`

from `mskt_hmm/EMMIXskew_dll/` inside the installed package.

There are two ways to obtain these libraries:

1. use the **prebuilt natives (Windows)**
2. build them locally from `src/mskt_hmm/EMMIXskew/src` (Windows or Linux)

### 2.a) Windows – use the prebuilt natives (recommended)

From the **repository root**:

```powershell
# ensure the package directory exists
New-Item -ItemType Directory -Force .\src\mskt_hmm | Out-Null

# extract prebuilt natives into src\mskt_hmm
tar -xf .\src\mskt_hmm\native_windows.zip -C .\src\mskt_hmm --strip-components=1
```

Expected layout after extraction (relative to the repo root):

```
src\mskt_hmm\EMMIXskew_dll\libemmixskew.dll
src\mskt_hmm\EMMIXskew_dll\libopenblas.dll
src\mskt_hmm\EMMIXskew_dll\libgfortran-5.dll
src\mskt_hmm\EMMIXskew_dll\libquadmath-0.dll
src\mskt_hmm\EMMIXskew_dll\libgomp-1.dll
src\mskt_hmm\EMMIXskew_dll\libgcc_s_seh-1.dll
src\mskt_hmm\EMMIXskew_dll\libwinpthread-1.dll
```

> Note: the ZIP may also contain additional files (e.g. an old `_hmmc*.pyd`);
> these are ignored by MsktHMM. What really matters is that
> `EMMIXskew_dll/libemmixskew.dll` and its dependencies end up under `mskt_hmm`.

You can test the loading of the library with:

```powershell
.\.venv\Scripts\Activate.ps1
py -c "import mskt_hmm.native as n; print('EMMIX lib loaded:', bool(n.LIB))"
```

If this prints `EMMIX lib loaded: True`, the native backend is ready.

#### 2.a.2) Windows – build EMMIXskew DLLs manually (only if prebuilt DLLs do not work)

If the prebuilt DLLs do not load on your setup (e.g. OpenBLAS mismatch), you
can rebuild `libemmixskew.dll` yourself using **MSYS2 UCRT64**.

1. Install toolchain in MSYS2 UCRT64:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc-fortran mingw-w64-ucrt-x86_64-openblas
```

2. Build from `src/mskt_hmm/EMMIXskew/src`:

```bash
cd /c/opt/workspace/python/MsktHMM/src/mskt_hmm/EMMIXskew/src   # adjust the path to your clone
rm -f *.o libemmixskew.dll libemmixskew.a

gfortran -O3 -fPIC -std=legacy -c \
  density2.f mixda.f mixmsn.f mixmsnda.f mixmst.f mixmstda.f \
  mixmvnda.f mixmvtda.f predmixdamsn.f predmixdamst.f scamstep.f

gcc -O3 -fPIC -std=c99 -DEMMIX_STANDALONE -Istandalone_rshim -c \
  standalone_rshim/special.c \
  density.c distance.c mixem.c mixinit.c estep.c mstep.c scaestep.c \
  module.c emmix_stubs.c

gcc -shared -o libemmixskew.dll *.o \
  -lopenblas -lgfortran -lquadmath -lm \
  -Wl,--export-all-symbols -Wl,--out-implib,libemmixskew.a

# copy next to the Python module
mkdir -p /c/opt/workspace/python/MsktHMM/src/mskt_hmm/EMMIXskew_dll
cp libemmixskew.dll /c/opt/workspace/python/MsktHMM/src/mskt_hmm/EMMIXskew_dll/
```

3. Make sure **all these DLLs** sit together with `libemmixskew.dll`:

```
libemmixskew.dll
libopenblas.dll
libgfortran-5.dll
libquadmath-0.dll
libgomp-1.dll
libgcc_s_seh-1.dll
libwinpthread-1.dll
```

You can inspect missing dependencies using `ntldd` (inside MSYS2):

```bash
ntldd -v /c/opt/workspace/python/MsktHMM/src/mskt_hmm/EMMIXskew_dll/libemmixskew.dll
```

---

### 2.b) Linux – build the `.so`

On Linux you build `libemmixskew.so` from the same sources.

Install the toolchain:

```bash
sudo apt update
sudo apt install -y gfortran libopenblas-dev build-essential
```

Build from `src/mskt_hmm/EMMIXskew/src`:

```bash
cd src/mskt_hmm/EMMIXskew/src
rm -f *.o libemmixskew.so

gfortran -O3 -fPIC -std=legacy -c \
  density2.f mixda.f mixmsn.f mixmsnda.f mixmst.f mixmstda.f \
  mixmvnda.f mixmvtda.f predmixdamsn.f predmixdamst.f scamstep.f

gcc -O3 -fPIC -std=c99 -DEMMIX_STANDALONE -Istandalone_rshim -c \
  standalone_rshim/special.c \
  density.c distance.c mixem.c mixinit.c estep.c mstep.c scaestep.c \
  module.c emmix_stubs.c

gcc -shared -o libemmixskew.so *.o -lopenblas -lgfortran -lm

mkdir -p ../../EMMIXskew_dll
cp libemmixskew.so ../../EMMIXskew_dll/
```

Test from Python:

```bash
python -c "import mskt_hmm.native as n; print('EMMIX lib loaded:', bool(n.LIB))"
```

---

## 3) Unpack test data

Some tests use pre-generated data stored as ZIP archives.

From the repository root (Windows PowerShell):

```powershell
Expand-Archive -Path .\src\mskt_hmm\tests\tests_MsktHMM\data_test\test_multi_state.zip `
               -DestinationPath .\src\mskt_hmm\tests\tests_MsktHMM\data_test -Force

Expand-Archive -Path .\src\mskt_hmm\tests\tests_MsktHMM\data_test\test_single_state_equivalence.zip `
               -DestinationPath .\src\mskt_hmm\tests\tests_MsktHMM\data_test -Force
```

On Linux:

```bash
unzip -o src/mskt_hmm/tests/tests_MsktHMM/data_test/test_multi_state.zip \
      -d src/mskt_hmm/tests/tests_MsktHMM/data_test

unzip -o src/mskt_hmm/tests/tests_MsktHMM/data_test/test_single_state_equivalence.zip \
      -d src/mskt_hmm/tests/tests_MsktHMM/data_test
```

---


## 4) Relationship with `hmmlearn`

This project **does not** vendor or modify `hmmlearn` anymore:

* `hmmlearn` is installed as a normal dependency via `pip`
* its C extension (`_hmmc*.pyd` / `_hmmc*.so`) comes from the official wheel
* MsktHMM only provides:

  * a new HMM class with uMST emissions (inside the `mskt_hmm` package)
  * the native EMMIXskew runtime (`libemmixskew`) used to compute uMST densities
    and expectations efficiently

You keep using `hmmlearn` as usual, and import the uMST HMM from `mskt_hmm`.

Example:

```python
from mskt_hmm.mskt_hmm import MsktHMM 

hmm = MsktHMM(
    n_components=3,        # number of hidden states
    n_features=d,          # dimension of the observation vectors
    # other params as in MsktHMM class
)

hmm.fit(X, lengths=lengths)
```

---

## 5) Run tests and demo

Activate your virtual environment and run:

```bash
# from project root
pytest src/mskt_hmm/tests/tests_MsktHMM
```

On Windows you can also use the convenience script:

```powershell
.\run_mskt_tests.bat
```

To experiment interactively, open the Jupyter notebook:

```bash
jupyter notebook mskt_hmm_demo.ipynb
```


---

## 6) Troubleshooting

If you see:

```text
RuntimeError: libemmixskew not uploaded/loaded
```

check that:

* `mskt_hmm/EMMIXskew_dll/libemmixskew.(dll|so)` exists in the installed package
  (or under `src/mskt_hmm/EMMIXskew_dll` when running from source);
* on Windows, all DLLs listed in section **3.a.2** are present in the same folder;
* you can successfully run:

  ```bash
  python -c "import mskt_hmm.native as n; print('EMMIX lib loaded:', bool(n.LIB))"
  ```

---

## 7) Notes

* EM for uMST follows **Sahu–Dey–Branco (2003)** (unrestricted skew-t),
  using the hierarchical representation and truncated-t expectations.
* The HMM layer reuses `hmmlearn`’s forward–backward and Viterbi core, so
  you get the usual API (`fit`, `score`, `predict`, `sample`, …).
* macOS is not officially supported here, but you can adapt the Linux build
  commands to produce a `.dylib` if you have a Fortran toolchain available.

```

