# Third-party sources and licensing boundaries

The root MIT license applies to this repository's original benchmark/orchestration code, copyright © 2026 Fereshteh Abolghasemi. It does not replace third-party licenses and does not establish commercial rights to all downloaded model components, weights, training datasets or sample images. No full third-party source tree, model weights or public/customer photo files are redistributed in this package. Setup obtains pinned upstream code/assets and preserves their existing license files; its compatibility patches do not relicense upstream code.

| Component | Primary source | Observed terms / action |
|---|---|---|
| HairFastGAN | https://github.com/AIRI-Institute/HairFastGAN | Repository LICENSE declares MIT, copyright 2024 AIRI. Preserve notices when redistributing its code. |
| HairFastGAN weights | https://huggingface.co/AIRI-Institute/HairFastGAN | Repository metadata declares MIT; this is not evidence that all inherited third-party assets have been relicensed. Download from source, not this repo. |
| SEAN | https://github.com/ZPdesu/SEAN | README declares CC BY-NC-SA 4.0 and academic research use only. A dependency of the hairstyle pipeline; commercial use needs separate resolution. |
| SegFormer skin/hair/clothing checkpoint | https://huggingface.co/isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing | Metadata declares MIT; model card is largely unfilled. Dataset, backbone and weights provenance must be reviewed separately for deployment. |
| CLIP | https://github.com/openai/CLIP | Preserve upstream code license and review terms for downloaded weights/data. |
| StyleGAN, e4e/pSp, FSE, ArcFace, BiSeNet, CtrlHair, HairCLIP, STAR and other inherited components | HairFastGAN's source/README references | Their source/weights/data can have separate terms. This list is an attribution inventory, not a completed legal clearance. |
| Public warmup samples | Pinned HairFastGAN weights repo, `input/6.png` and `input/7.png` | Downloaded at runtime, not bundled here. Public availability is not itself permission to republish photos. |
| Python packages | Frozen installed requirements | Preserve their terms if redistributing environments; environments are not bundled. |

Checked primary project pages during preparation on 2026-10-05. This is a release notice, not a claim of legal review or an endorsement by the original authors. Research publication of the independent harness should not be described as creation of a new GAN or unrestricted commercial licensing of the whole pipeline.

HairFastGAN code revision: `49e98019c9243d1991c19d45764fc3340a498de2`.
Weights revision: `47855f8ea72b02294649385211a1ae1fd7dcb58e`.
SegFormer revision: `dd02ee5b73017377ba6f12a58c3e300cbbfd58c8`.
CLIP code revision: `a1d071733d7111c9c014f024669f959182114e33`.

Research citation:

```bibtex
@article{nikolaev2024hairfastgan,
  title={HairFastGAN: Realistic and Robust Hair Transfer with a Fast Encoder-Based Approach},
  author={Nikolaev, Maxim and Kuznetsov, Mikhail and Vetrov, Dmitry and Alanov, Aibek},
  journal={arXiv preprint arXiv:2404.01094},
  year={2024}
}
@InProceedings{Zhu_2020_CVPR,
  author={Zhu, Peihao and Abdal, Rameen and Qin, Yipeng and Wonka, Peter},
  title={SEAN: Image Synthesis With Semantic Region-Adaptive Normalization},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
