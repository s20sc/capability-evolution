# Capability evolution in embodied agents

Materials for **Capability evolution in embodied agents: A controlled study of skill acquisition, retention, and repair**, by Xue Qin, Simin Luan, Cong Yang and Zhijun Li.

## Paper and reproduction

- [Manuscript with figures](paper/manuscript.pdf) and [editable source](paper/main.tex)
- [Reproduction guide](study/README.md)
- [Tables and figures to saved-result mapping](study/DATA_GUIDE.md)
- [Drift-selection procedures and all nine saved scores](supplements/drift-selection/README.md)
- [Workflow](study/WORKFLOW.md), [operators](study/OPERATORS.md), [data preparation](study/DATA_PREPARATION.md) and [experience reconstruction](study/EXPERIENCE.md)

The three experiments concern acquisition, complete-parent learning paths and corrective experience. They use prescribed learning operations in simulation; autonomous operation selection and recursive self-improvement were not evaluated. Gains, retained successes and repaired failures are reported separately.

Software and saved observations are included. Third-party demonstrations, native runtime assets, generated-experience arrays, starting states and recorded-transition companion archives are separate inputs documented in the guides. The companion materials currently have confidential peer-review access; a public archive identifier will be linked upon release.

## Version and earlier work

This snapshot accompanies manuscript v30. The earlier governance-focused project is preserved by the `legacy-jss-v16` Git tag and its September 2026 commit history. Its preprint is [arXiv:2604.07799v3](https://arxiv.org/abs/2604.07799v3); it is a different study and is not the current manuscript.

## Licensing

Original software is Apache-2.0; original documentation and measurements use CC BY 4.0. See [license boundaries](study/LICENSE_SCOPE.md) and [third-party notices](study/THIRD_PARTY_NOTICES.md). Before restoring learned states, read the trusted-input and isolation guidance in the reproduction guide.
