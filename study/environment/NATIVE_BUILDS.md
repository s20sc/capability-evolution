# Rebuilding the three locally built dependency inputs

The recorded environment used 64 published wheel files and three locally built wheels: egl-probe 1.0.2, evdev 2.0.0 and gym 0.26.2. The unmodified original source distributions for these three builds are supplied in source-distributions/, including their original notices. native-build-inputs.json records their source URLs, sizes and SHA-256 identities, the successful build interfaces and the original build-tool evidence. No source or dependency code was executed when assembling this companion.

The applicable licenses are MIT for egl-probe and gym, and BSD-3-Clause for evdev. Each original license remains inside its source archive.

## Build and runtime environments are distinct

Use Linux x86_64 with CPython 3.11.16 for the recorded native wheel target. Provide a C/C++ compiler, make, Python development headers and Linux input headers for evdev. The evdev source checks linux/input.h, linux/input-event-codes.h and linux/uinput.h; generated event codes can depend on the installed headers. egl-probe uses CMake to compile its bundled GLAD sources and links dl and pthread. At runtime its native device discovery still requires a compatible EGL/graphics installation; successful compilation alone does not establish that device access works.

The recorded evdev build used pip 26.2.1, setuptools 80.10.2 and wheel 0.48.0, with packaging 24.2 in a separate build-only layer. The experiment runtime retained packaging 23.0. With the older packaging version in the build environment, evdev metadata preparation failed; the successful build used the separate 24.2 layer. Do not interpret the build-only package as a changed experiment dependency. The exact packaging 24.2 wheel URL and identity are included in native-build-inputs.json.

The egl-probe build used CMake 4.1.3. Its source requests an older minimum CMake policy; the recorded build set CMAKE_POLICY_VERSION_MINIMUM=3.5 without editing the source. This setting does not certify all future CMake/compiler combinations.

## Build commands

Verify the three source archive hashes against SOURCE_IDENTITIES.json before extracting them into fresh source directories. Prepare a separate build interpreter with the build tools listed above; obtain its tools before the offline build step. Let BUILD_PY be its absolute Python path and WHEELHOUSE an absolute destination directory.

From the extracted egl_probe-1.0.2 source root:

```sh
CMAKE_POLICY_VERSION_MINIMUM=3.5 PIP_NO_INDEX=1 \
  "$BUILD_PY" setup.py bdist_wheel --dist-dir "$WHEELHOUSE"
```

For gym and evdev, pass their extracted source directories:

```sh
"$BUILD_PY" -m pip wheel --no-deps --no-build-isolation --no-index \
  --wheel-dir "$WHEELHOUSE" /absolute/source/gym-0.26.2
"$BUILD_PY" -m pip wheel --no-deps --no-build-isolation --no-index \
  --wheel-dir "$WHEELHOUSE" /absolute/source/evdev-2.0.0
```

The build interpreter must already contain its tooling. The commands do not download dependencies. Do not select Gym's optional MuJoCo extras: the experiment's MuJoCo version is the separately pinned 2.3.2 input, not the version named by an unrelated Gym extra.

## Installing a newly built runtime

Download the 64 published wheel URLs listed in wheel-sources.json and verify their hashes. With a separate runtime interpreter, install those fixed inputs from the local wheelhouse:

```sh
"$RUNTIME_PY" -m pip install --no-index --no-deps --require-hashes \
  --find-links "$WHEELHOUSE" -r requirements64-published-wheels.txt
```

Record the three newly built wheel hashes in a separate local manifest, then install those wheels without resolving additional dependencies. Retain requirements67-with-hashes.txt as the identity of the original environment inputs; its three local-wheel hashes are not promised to match a new build. Compiler paths, archive timestamps, generated header data and build metadata can change wheel bytes. In particular, the original egl-probe wheel included local CMake build-cache records and is not distributed here as a clean, universal binary.

The separate fixed runtime-source companion supplies the exact robomimic/robosuite package bodies and assets. Combine it with the wheel target as described in DEPENDENCIES.md. Verify the required source hashes, versions and device/graphics conditions before running learning or evaluation. Rebuilding these dependencies on a different host is distinct from the reported same-host installation and the bounded learning/evaluation checks; an independent complete environment or training replication is not claimed.
