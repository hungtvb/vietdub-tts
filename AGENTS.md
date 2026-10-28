# VietDub native TTS development workflow

Follow the requested [Superpowers skills](https://github.com/obra/superpowers): `using-superpowers`, `brainstorming`, `writing-plans`, `using-git-worktrees`, `test-driven-development`, `verification-before-completion`, and `requesting-code-review`. For Codex or compatible agents, install the actual upstream skills using the provider's supported method (e.g. `npx skills add https://github.com/obra/superpowers --skill using-superpowers`); **reading this file does not mean skills were automatically installed**.

- New architectural work: confirm the design, write the spec and executable plan under `docs/superpowers/`, and use a feature branch/PR; do not commit directly to `main`.
- Every change: write a failing test, verify RED, implement the smallest change, verify GREEN, then run the entire repository suite and inspect the actual results.
- TTS acceptance: a genuinely **intelligible unseen Vietnamese sentence**, not CI success, waveform RMS, non-DC signal or synthetic fixture.
- Vocoder acceptance: held-out real-audio reconstruction listened to against source by a reviewer with appropriate voice/data rights.
- No pretrained TTS dependencies or made-up labels/consent; preserve data provenance, source separation and private checkpoints.
- Read `docs/superpowers/specs/2026-10-26-native-vocoder-quality-design.md` and linked plan before implementing the next vocoder milestone.
