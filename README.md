# PadLab — OpenCV image processing lab

App C++17, Qt5 Widgets và OpenCV để thử nghiệm pipeline phát hiện và phân vùng đối tượng. Xử lý ảnh hoàn toàn local, không dùng model/AI, không upload ảnh.

## Build bằng Docker Ubuntu 22.04

```bash
./scripts/build-docker.sh
```

Docker build compile bằng GCC 11, Qt 5.15 và OpenCV 4.5.4, chạy kiểm thử engine, GUI offscreen và CLI. Image runtime: `padlab:ubuntu22`. Binary và demo xuất vào `artifacts/ubuntu22/`. Docker cần quyền truy cập daemon; nếu terminal chưa có quyền, chạy lệnh Docker bằng cơ chế quản trị của máy.

`.dockerignore` chỉ đưa source và tests vào build context; không đưa dataset hoặc thư mục artifacts vào image.

## Mở giao diện trên desktop hiện tại

```bash
./scripts/run-gui.sh /data/artifacts/ubuntu22/demo/images/pad-0.png
```

Script dùng Wayland nếu có, hoặc X11. Chạy bằng UID hiện tại, chỉ mount project tại `/data`, không cần `xhost +`, tắt network cho container. Trên Fedora, `label=disable` cho phép truy cập socket desktop mà không đổi nhãn SELinux của dữ liệu. Nếu dữ liệu ở ngoài project, copy app vào công ty và mount thư mục cần dùng tại `/data`, hoặc chạy binary native Ubuntu. Với X11, cần Xauthority hợp lệ của session.

## Chạy native Ubuntu 22.04 x64

Copy `artifacts/ubuntu22/` sang máy đích. Đây là binary liên kết động, không phải bộ bundle mọi thư viện:

```bash
sudo apt-get install libqt5widgets5 libqt5concurrent5 libopencv-core4.5d \
  libopencv-imgproc4.5d libopencv-imgcodecs4.5d qtwayland5
./bin/padlab
```

Hoặc mang image Docker offline bằng `docker save -o padlab-ubuntu22.tar padlab:ubuntu22`, rồi `docker load -i padlab-ubuntu22.tar` tại máy đích. Các thao tác trên dữ liệu không cần Internet; tải dependency/build image ban đầu cần Internet hoặc mirror nội bộ.

## Sử dụng

1. **Open image**: ảnh gốc bên trái; overlay, mask hoặc kết quả từng bước bên phải. Wheel để zoom, kéo để pan, **Fit both** để fit.
2. Thêm bước từ dropdown, bật/tắt bằng checkbox, đổi thứ tự với ↑/↓, sửa tham số. Preview chạy nền, debounce 250 ms. Config chưa hợp lệ hiển thị lỗi và xóa điểm cũ.
3. **Load GT mask / Load GT boxes** để đánh giá một ảnh. **Clear GT** để bỏ nhãn hiện tại. Mở ảnh mới sẽ xóa GT cũ.
4. **Folder evaluation**: chọn folder ảnh, mask và/hoặc boxes; chọn một thư mục output MỚI. Folder scan không recursive. **Run folder**, **Cancel**, lọc ảnh chưa đạt và double-click để xem lại với config của batch đó.
5. **Export pipeline** lưu JSON phiên bản 1, dùng lại trong GUI hoặc CLI. **Export result** xuất mask, overlay, boxes, metrics, config và crop pad kèm mask crop.

Mỗi batch dùng snapshot config lúc bấm Run; chỉnh config trong lúc chạy không thay đổi batch đang xử lý. Ảnh lỗi được ghi thành `error` và batch tiếp tục. Hủy thực hiện giữa bước OpenCV/ảnh; một lời gọi OpenCV đang chạy phải kết thúc trước. Output cũ và dataset không bị ghi đè; đường dẫn output phải tách khỏi folder ảnh/nhãn. Giới hạn 2.000 vùng/ảnh để tránh xuất crop và matching quá lớn; lọc nhiễu bằng components trước.

Các bước: grayscale; kênh BGR/HSV/Lab; CLAHE; Gaussian/median/bilateral; fixed/Otsu/adaptive threshold; HSV range; Canny; morphology; invert; fill holes; lọc connected components theo area, aspect ratio, extent và largest-only. Không resize ảnh trong pipeline.

Tham số enum hiển thị ở panel trái. Kernel/block phải lẻ. `channel.space`: 0 BGR, 1 HSV, 2 Lab; chỉ số channel 0..2. Hue HSV 0..179. Morphology: 0 open, 1 close, 2 erode, 3 dilate, 4 top-hat, 5 black-hat. Shape: 0 rectangle, 1 cross, 2 ellipse. Blur sau mask làm ảnh mất tính nhị phân nên cần threshold lại trước khi lọc component hoặc xuất.

## Ground truth

Với `images/photo.jpg`:

```text
masks/photo.jpg.png
boxes/photo.jpg.json
```

Dùng đầy đủ tên file để `photo.jpg` và `photo.png` không trùng nhãn/output. Mask PNG **8-bit, một kênh**, kích thước đúng ảnh gốc: 0 là background, mọi giá trị khác 0 là foreground. App không tự resize GT. Box dùng pixel ảnh gốc, axis-aligned, `[x, y, width, height]`, biên phải/dưới exclusive:

```json
{"schema_version": 1, "boxes": [[100, 80, 300, 150]]}
```

`boxes: []` biểu thị ảnh đã gán nhãn không có pad. Thiếu file GT được coi là **unlabeled**, không được tính điểm. GT lỗi bị loại khỏi aggregate và ghi lỗi. Nếu có mask mà không có boxes, detection GT suy ra từ connected components 8-connectivity của mask. Nếu pad chạm nhau, cần cung cấp box GT riêng vì binary mask không có instance ID. Box GT riêng được ưu tiên cho detection; mask GT vẫn dùng cho pixel metrics.

## Metrics và mục tiêu >99%

- Mask: IoU, Dice, foreground precision/recall, pixel accuracy, TP/FP/FN/TN. Accuracy gồm background nên có thể cao khi mask sai; ưu tiên IoU/Dice.
- Box: ghép một-một maximum-cardinality với IoU ≥ ngưỡng (mặc định 0.99), TP/FP/FN, precision/recall/F1 và IoU trung bình các cặp matched. Box trùng bị tính FP. Không tính mAP: pipeline không sinh confidence score.
- Pass: nếu có mask thì **mask IoU > target**; nếu có detection GT thì không được FP/FN tại ngưỡng box đã đặt. Có cả hai phải đạt cả hai. Mask target mặc định 0.99.
- Hai mask rỗng có IoU/Dice = 1; mask predict rỗng nhưng GT có foreground thì IoU = 0. Khi mẫu số precision = 0, trả 0 nếu có FN, ngược lại 1; recall của GT rỗng = 1. Không có GT: metrics = null/N/A.
- Aggregate mask gồm macro IoU (mỗi ảnh trọng số bằng nhau) và micro IoU/Dice (cộng pixel). Detection aggregate cộng TP/FP/FN. Các ảnh rỗng đã gán nhãn được tính vào macro; nên xem riêng ảnh có pad khi kết luận chất lượng.

Output: `pipeline.json`, `report.csv`, `metrics.json`, `summary.json`, `masks/`, `overlays/`, `boxes/`, `crops/`. Thời gian `ms` chỉ là pipeline + overlay/boxes, không bao gồm decode ảnh, đọc GT hoặc ghi file. Thông tin ảnh/đường dẫn nhãn chỉ lưu trong output local.

**Mục tiêu IoU >99% cần được xác nhận trên dữ liệu thực tế.** Demo tổng hợp chỉ kiểm tra chức năng. Chốt pipeline trên tập tuning, rồi đánh giá trên tập holdout khác lô/ánh sáng và kiểm tra các ảnh lỗi; không lấy chính tập đã tinh chỉnh để kết luận độ chính xác.

## CLI: sử dụng lại pipeline đã xuất

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --security-opt label=disable -v "$PWD:/data" padlab:ubuntu22 \
  padlab-cli --config /data/artifacts/ubuntu22/demo/pipeline.json \
  --input /data/artifacts/ubuntu22/demo/images \
  --masks /data/artifacts/ubuntu22/demo/masks \
  --boxes /data/artifacts/ubuntu22/demo/boxes \
  --output /data/results-demo
```

`--write-default pipeline.json` xuất config mặc định. Exit code: 0 chạy xong, 1 lỗi setup/config, 2 có ảnh/GT lỗi, 130 bị hủy. Ảnh `below_target` là kết quả đánh giá hợp lệ nên vẫn exit 0; đọc report để quyết định pass sản xuất. CLI dùng cùng engine với GUI; có thể tái sử dụng `src/pipeline.*` trong app C++ (QtCore để đọc JSON; các thuật toán xử lý là OpenCV).

## Kiểm thử / bố cục source

`src/pipeline.*`: schema, validation, các bước và metrics. `src/dataset.*`: pairing, batch và export. `src/window.*`: giao diện. `tests/`: engine, GUI và CLI, dữ liệu hoàn toàn tổng hợp. Kiểm thử GUI render qua Qt offscreen; screenshot trong `artifacts/ubuntu22/gui-preview.png`. Docker build chạy CTest và dừng nếu test thất bại.

Các API contour/connected-components được dùng theo [tài liệu OpenCV](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html).


## Xem từng bước và đo biên chính xác

Bấm trực tiếp một bước trong pipeline để xem kết quả, dùng **Previous step / Next step** để chuyển. **Before / after step** đặt đầu vào của bước bên trái và đầu ra bên phải. Zoom được giữ khi thay đổi tham số hoặc chuyển view cùng kích thước. Bước tắt được báo rõ, không hiển thị nhầm output bước khác. **Export all steps** xuất ảnh PNG theo số thứ tự, config, thời gian và manifest; các ảnh overlay đã được vẽ biên nếu có.

Menu **Presets** có **Precision boundaries** và **Low contrast + boundaries**. Đây là điểm bắt đầu để thử, không phải cấu hình đã tối ưu cho một dataset cụ thể. `images.jpeg` chỉ là ảnh xem thử local, không được đưa vào repo hay release.

| Bước | Chức năng / tham số chính |
| --- | --- |
| `illumination` | Ước lượng nền bằng Gaussian; `divide=0`: trừ nền +128; `divide=1`: chia nền. `sigma` nên lớn hơn cấu trúc cần giữ. |
| `guided` | Self-guided filter giữ biên, với `radius` và `epsilon` trên cường độ chuẩn hóa 0..1. |
| `sauvola` | Threshold theo mean/std cục bộ: `T = mean * (1 + k * (std / range - 1))`. Chỉnh block, k, range và inverse. |
| `watershed` | Distance transform tạo seed theo cực đại cục bộ rồi marker-controlled watershed để thử tách vùng chạm nhau. Kết quả phụ thuộc seed và có thể làm mất một dải pixel ở ranh giới. |
| `edge_localize` | Từ contour mask cuối, tìm đỉnh gradient dọc pháp tuyến trong bán kính `radius`. Gaussian `sigma`, ngưỡng `min_gradient`, `normal_span` chỉnh mức ổn định. Source 0 = grayscale ảnh gốc; source 1 = ảnh cường độ một kênh được lưu gần nhất. |
| `subpixel` | Nội suy parabol qua ba mẫu quanh đỉnh gradient để tinh chỉnh vị trí tối đa ±0.5 pixel. `min_curvature` loại đỉnh phẳng. Phải nằm ngay sau `edge_localize` trong các bước bật. |

Các bước xử lý mask phải nằm trước hai bước đo biên. Cả contour ngoài và lỗ trong được giữ cùng quan hệ parent. Điểm gradient yếu, peak nằm ngoài phạm vi tìm kiếm hoặc thiếu mẫu gần mép ảnh giữ vị trí seed và được đánh dấu rejected. Cyan là biên đo, đường đứt là seed, cam là điểm không định vị được. Giao diện vẽ vector ở tọa độ float để xem khi zoom, JSON giữ tọa độ float cùng trạng thái localized/refined và strength, thêm min-area rotated rectangle cho từng contour.

`boundaries.json` (ảnh đơn) và `boundaries/<tên ảnh>.json` (batch) dùng tọa độ ảnh gốc: x là cột, y là hàng, tọa độ nguyên là tâm pixel. **Đây là phép đo biên; không tự sửa binary mask hoặc box nguyên.** Vì vậy IoU mask/box vẫn đánh giá đầu ra phân vùng, không phải phép đo sai số subpixel. Độ chính xác subpixel cần kiểm tra trên ảnh có biên chuẩn và điều kiện chụp phù hợp; không thể suy ra từ nhiều chữ số thập phân trong JSON.

Guided filter và Sauvola được hiện thực từ các phép toán OpenCV core/imgproc, không thêm dependency opencv-contrib. Tham khảo [Guided Image Filtering](https://kaiminghe.github.io/publications/eccv10guidedfilter.pdf), [Sobel derivatives](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html), [Sauvola trong tài liệu OpenCV](https://docs.opencv.org/4.x/df/d2d/group__ximgproc.html).

## Binary release

Mỗi release có `.deb`, `.tar.gz` cho **Ubuntu 22.04 x86_64** và `SHA256SUMS`. Cài bản `.deb` bằng `sudo apt install ./padlab-0.1.0-ubuntu22.04-amd64.deb`; lệnh này giải quyết các thư viện runtime. Sau đó mở **PadLab** từ menu ứng dụng hoặc chạy `padlab` trong terminal. Gói tar chứa binary liên kết động; xem `INSTALL.txt` để cài dependency.

Build local bằng `./scripts/build-docker.sh`; gói release nằm trong `artifacts/ubuntu22/release/`. Cả GUI/CLI và các thuật toán mới đều được kiểm thử trong Docker Ubuntu 22.04 trước khi đóng gói.


Để tự push và tạo GitHub Release sau khi đăng nhập:

```bash
./scripts/publish-release.sh gabapppp/padlab
```

Script kiểm tra checksum và tag, đăng nhập GitHub CLI nếu cần, tạo repo public nếu chưa tồn tại, push nhánh `main` + tag `v0.1.0`, rồi upload ba file release. Script không thay thế release hoặc tag đã tồn tại.
