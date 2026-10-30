# Dependencies and third-party materials

## Fixed native sources

The separate P2-runtime-sources.zip companion supplies the exact robomimic 0.3.0 and robosuite 1.4.1 package sources and native assets used with the version-pinned Python stack. Its SOURCE_IDENTITIES.json records every supplied source/asset file. Original MIT notices and MuJoCo Apache/third-party notices accompany the associated components. Retain their applicable terms; the study's license does not replace them.

The software licenses are available from the upstream [robomimic v0.3.0](https://github.com/ARISE-Initiative/robomimic/blob/v0.3.0/LICENSE) and [robosuite v1.4.1](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/LICENSE) releases. Source-package versions alone do not replace the supplied source identities. The companion excludes local installation metadata, caches, the Python interpreter, wheel bodies, operating-system libraries and graphics drivers.

environment/requirements67-with-hashes.txt records the wheel bytes used by the tested target; it is not a universal lock for every platform. In particular, native-built egl-probe and evdev wheels may depend on the build environment. environment/source-identities.json supplies the additional bound source identities. The measured stack uses Linux CPython 3.11.16, torch 2.7.0+cu128, numpy 1.26.0, MuJoCo 2.3.2 and an RTX 3090. Learning requires the configured CUDA runtime; CPU task evaluation still needs compatible native MuJoCo/EGL/OpenGL and system libraries. A new installation must establish those prerequisites; the source companion does not certify an arbitrary OS or driver.

The original source distributions for the three locally built wheel inputs (egl-probe, evdev and gym) are included under environment/source-distributions/, with provider URLs, original hashes and notices. environment/NATIVE_BUILDS.md gives the recorded build interfaces, the separate packaging 24.2 build layer and the packaging 23.0 runtime distinction. environment/requirements64-published-wheels.txt retains the 64 published wheel hashes. A new build has its own wheel hashes; the original 67-wheel list remains the as-run identity record.

## Provider demonstration files

Obtain these exact files from the provider; they are not included in the study-source companion:

| Input | Provider URL | SHA-256 |
|---|---|---|
| Can raw states | https://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/demo_v141.hdf5 | 8b811d652f21b0b7a3b16fb8407521963a974311260a4e9ffc40600d08e6b3cf |
| Square low-dimensional | https://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/low_dim_v141.hdf5 | afc1fc1bd72193d1fd89c689b738cb9eb1a17352217aea2a2e029ccc49eefcbd |
| Square raw correspondence input | https://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/demo_v141.hdf5 | 53f202c7d6dfc91a4a30964ed7ce63ecd8a4f67818fa02cf0d251e6aeddfc586 |

The [release-specific provider documentation](https://robomimic.github.io/docs/v0.3/datasets/robomimic_v0.1.html) describes the v1.4.1 route. The current provider release is not a byte-identical substitute. The software MIT notice is not represented here as an independently verified license for these separately downloaded demonstration objects; retain applicable provider terms and attribution.

## Generated experience and learned states

The separate experience and state companions carry their exact input identities and scientific roles. They do not include the provider demonstration files or a full third-party runtime. The study used MimicGen during experience preparation; its [fixed-commit documentation](https://github.com/NVlabs/mimicgen/blob/c0fbff22ba40060f05a917a80817f3f5c20731e7/README.md) distinguishes its NVIDIA source-code license from the CC BY 4.0 license for its released datasets. That statement is not used to relicense the robomimic demonstrations, this study's outputs or all linked code. MimicGen is not imported by the downstream learning and evaluation entries supplied here; regenerating its upstream experience requires its separately specified code and inputs.

The fixed MimicGen source companion supplies its original package/templates and NVIDIA license. experience/vendor-support preserves the original gdown/BeautifulSoup/SoupSieve/PySocks sources and notices; experience/generation-import-requirements.txt lists additional import-support pins separately from the 67-wheel learning stack. The recorded-transition companion supplies the original target/correction episodes and five exposed Square100000 parent logs for reconstruction. These supplied objects do not relicense provider demonstrations or certify a complete independent generation runtime.

## Trust and execution boundary

Study and dependency sources are executable software, and the learned-state companion uses full-object PyTorch serialization rather than a weights-only interchange format. Verify origins and release identities independently before running or loading them. Checksums detect differences from the selected manifest; they are not a malware analysis or a substitute for a trusted source. Full restoration must run in an isolated, least-privilege environment without unrelated credentials or sensitive files. The restoration checks and supported CUDA requirements must not be relaxed to accommodate unknown input files. Provider demonstrations and third-party source archives retain their own terms and trust requirements.
