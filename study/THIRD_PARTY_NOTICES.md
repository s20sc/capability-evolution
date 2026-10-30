# Third-party software notices

Third-party files retain their original licenses. The study's root software license does not replace these terms. This index points to the accompanying license texts; those texts govern each component.

## Components included in the study-source package

| Component | Version or scope | License | Accompanying notice |
|---|---|---|---|
| robomimic compatibility code | robosuite environment compatibility module in experience/compat_env_robosuite.py | MIT; retain the upstream notice for the incorporated code | experience/LICENSE.robomimic.txt |
| Beautiful Soup | 4.12.3; experience/vendor-support/bs4 | MIT | experience/vendor-support/beautifulsoup4-4.12.3.dist-info/licenses/LICENSE |
| gdown | 5.2.0; experience/vendor-support/gdown | MIT | experience/vendor-support/gdown-5.2.0.dist-info/licenses/LICENSE |
| Soup Sieve | 2.6; experience/vendor-support/soupsieve | MIT | experience/vendor-support/soupsieve-2.6.dist-info/licenses/LICENSE.md |
| PySocks | 1.7.1; experience/vendor-support/socks.py and sockshandler.py | BSD, three conditions | experience/vendor-support/PySocks-1.7.1.dist-info/LICENSE |
| egl-probe source distribution | 1.0.2 | MIT | environment/source-distributions/egl_probe-1.0.2.tar.gz: egl_probe-1.0.2/LICENSE |
| evdev source distribution | 2.0.0 | BSD, three conditions | environment/source-distributions/evdev-2.0.0.tar.gz: evdev-2.0.0/LICENSE |
| Gym source distribution | 0.26.2 | MIT | environment/source-distributions/gym-0.26.2.tar.gz: gym-0.26.2/LICENSE.md |

The original source-distribution archives are unchanged. Their complete notices remain inside the archives. The metadata and identities supplied with the support files describe their original versions.

## Separate runtime and generation companions

P2-runtime-sources.zip contains fixed robomimic and robosuite sources and native assets. Its third-party-notices/ directory contains robomimic-v0.3-LICENSE.txt, robosuite-v1.4.1-LICENSE.txt, mujoco-2.3.2-LICENSE and mujoco-2.3.2-LICENSES_THIRD_PARTY.md. Retain those notices with their corresponding components, including the listed asset and nested-library terms.

P2-MimicGen-sources.zip contains the original NVIDIA License in LICENSE. Section 3.1 sets its redistribution conditions, and section 3.3 restricts use to research or evaluation. This companion is not relicensed under Apache 2.0 or Creative Commons. These source-code terms are not represented as a blanket license determination for every generated output.

Other installed dependencies are identified in environment/ and DEPENDENCIES.md. Package pins and download locations do not replace those dependencies' own licenses. Provider demonstrations are not bundled with the study-source package.
