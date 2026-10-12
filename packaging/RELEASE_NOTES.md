PadLab v0.2.0 is a local Qt5 + OpenCV workbench for configuring, inspecting and evaluating image processing pipelines.

- Broader image import through OpenCV and Qt image plugins, including BMP, high-depth TIFF, JPEG 2000, HDR/PFM and uncommon raster formats. Input conversion metadata is saved in reports.
- Auto contrast, gamma, Otsu/Triangle threshold with foreground polarity, and gradient-based Canny; selected values are inspectable and exported.
- Three automatic starting presets, drag-and-drop, folder image navigation, named parameter controls, and pipeline undo/redo.
- Single-image previews and folder evaluation with masks, boxes, IoU, Dice, precision/recall and CSV/JSON reports.
- Click-to-inspect pipeline steps, before/after views, previous/next controls and step-image export.
- Illumination correction, guided filtering, Sauvola thresholding and marker-controlled watershed.
- Normal-profile edge localization and subpixel boundary refinement with vector overlays and floating-point JSON export.
- GUI and CLI share the same validated pipeline configuration.

Downloads target **Ubuntu 22.04 x86_64**. Install the Debian package with:

```bash
sudo apt install ./padlab-0.2.0-ubuntu22.04-amd64.deb
padlab
```

The tar.gz contains dynamically linked binaries; see INSTALL.txt for runtime dependencies. SHA256SUMS covers both packages.

Built and tested in an Ubuntu 22.04 Docker environment with Qt5 and OpenCV 4.5.4. The Docker build runs six CTest suites. Boundary measurements do not alter the raster mask or integer-box metrics. The workspace remains 8-bit BGR; multi-page/animated files use their first frame. Test images are synthetic; accuracy on a particular dataset requires its ground truth.
