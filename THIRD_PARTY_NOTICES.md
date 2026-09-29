# Third-Party Notices

## FreeFlow

GlitchScribe's voice Edit Mode product pattern, collision-safe hold/toggle
shortcut sessions, global Escape cancellation behavior, and the integration
analysis in `docs/glitchscribe/research/2026-08-03-freeflow-integration-audit.md` were
informed by [FreeFlow](https://github.com/zachlatta/freeflow), audited at commit
`e63a232d088528bf86a8b28ea18ab7916ef84895`.

FreeFlow is distributed under the following license:

MIT License

Copyright (c) 2026 Zach Latta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## NVIDIA Nemotron 3.5 ASR Streaming 0.6B

GlitchScribe can use the user-installed
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
checkpoint as its primary local speech-recognition model. The repository does
not redistribute the checkpoint. The local model manifest pins upstream
revision `1c8deaecc64b91f034d73e08dd8b64625eb3395d` and official weights SHA-256
`9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174`.

The checkpoint is made available under the
[Open Model Development and Weights License 1.1](https://openmdw.ai/license/1-1/).

## PyTorch, Transformers, and Safetensors

Nemotron inference uses:

- [PyTorch](https://pytorch.org/) — BSD 3-Clause license.
- [Transformers](https://github.com/huggingface/transformers) — Apache License 2.0.
- [Tokenizers](https://github.com/huggingface/tokenizers) — Apache License 2.0.
- [Safetensors](https://github.com/huggingface/safetensors) — Apache License 2.0.
- [Librosa](https://github.com/librosa/librosa) — ISC license.

The full dependency license texts are distributed with their installed Python
packages. These libraries are lazy-imported only when the explicitly selected
Nemotron model is loaded.
