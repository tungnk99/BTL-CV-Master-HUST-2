Với ảnh CIFAR (32×32) **BoVW cổ điển vẫn có thể làm được**, nhưng vì ảnh quá nhỏ bạn phải **thay đổi cách trích đặc trưng** (dense patches / upsampling / descriptor học được từ CNN) và **dùng các phương pháp tổng hợp mạnh hơn** (VLAD / Fisher / Spatial Pyramid) + chuẩn hoá tốt. Nếu mục tiêu là “kết quả tốt nhất”, hướng mạnh mẽ hiện nay là **dùng feature map từ CNN tiền huấn luyện làm descriptors rồi aggregate bằng VLAD/FV + SVM tuyến tính** — thường vượt trội so với SIFT/ORB trên ảnh nhỏ như CIFAR.

Dưới đây là pipeline chi tiết, kinh nghiệm và các giá trị tham khảo để bạn thử nghiệm.

# 1) Pipeline đề xuất (từ đơn giản → mạnh)

1. **Tiền xử lý**

   * Upscale mỗi ảnh từ 32×32 → **64×64** hoặc **96×96** (bilinear). Lý do: trình phát hiện/descriptor như SIFT cần kích thước đủ lớn để có mẫu có ý nghĩa; upscaling không tạo thông tin mới nhưng giúp descriptor ổn định hơn.
   * Hoặc giữ 32×32 nếu dùng CNN conv-features (thường ổn với kích thước nhỏ).
   * Chuẩn hoá ảnh: convert sang grayscale cho SIFT; với CNN dùng RGB và theo chuẩn của CNN (mean/std normalization).

2. **Trích đặc trưng (descriptor) — chọn 1 trong các cách**

   * **Dense SIFT** (nếu dùng BoVW cổ điển): dense sampling (ví dụ patch step = 4 px, patch size = 8–16). Tránh chỉ dùng keypoint detector vì ảnh quá nhỏ thiếu keypoints ổn định.
   * **ORB/BRIEF**: nhanh, nhược điểm: thường kém so với SIFT trên biểu diễn texture.
   * **CNN intermediate features (khuyến nghị)**: dùng một CNN tiền huấn luyện (ResNet18/VGG16) → lấy feature maps của 1–2 layer giữa (ví dụ conv3_x) → mỗi spatial vector (ví dụ 8×8×C) coi là descriptor. Đây gần như luôn cho biểu diễn tốt hơn cho ảnh nhỏ.
   * **Kết hợp màu**: dùng descriptors trên từng kênh (opponent-SIFT) hoặc thêm thông tin màu nếu cần.

3. **Giảm chiều & chuẩn bị cho clustering**

   * Áp PCA lên descriptor (ví dụ SIFT 128 → PCA 64) để giảm noise & tăng tốc KMeans.
   * Lấy mẫu ngẫu nhiên tối đa ~1–5 triệu descriptors từ toàn bộ tập train để build codebook.

4. **Xây codebook (vocabulary)**

   * **MiniBatchKMeans** (sklearn) hoặc Faiss nếu lượng lớn.
   * **K (vocab size)**: thử 256, 512, 1024. Với ảnh nhỏ, 256–512 thường đủ; quá lớn dễ noise/overfit.
   * Với CNN-descriptors có thể dùng K từ 64–512 vì mỗi descriptor giàu thông tin.

5. **Tạo biểu diễn ảnh (encoding / pooling)**

   * **Hard-assignment BoW histogram** (đếm) — baseline.
   * **Soft assignment** (gaussian weighting) — cải thiện nhẹ.
   * **VLAD** hoặc **Fisher Vector (FV)** — mạnh hơn nhiều; nếu muốn kết quả tốt nhất, thử VLAD/FV + PCA + power-normalization.
   * **Spatial Pyramid Matching (SPM)**: chia ảnh theo lưới (1×1, 2×2) và concatenation histogram/VLAD cho giữ chút thông tin vị trí.

6. **Chuẩn hoá vector đặc trưng**

   * L2-normalize; hoặc **signed square-root (power) + L2** (a.k.a Hellinger kernel transform) — rất quan trọng khi dùng histograms/VLAD.
   * Nếu dùng FV, thường áp dụng power-normalize rồi L2.

7. **Classifier**

   * **Linear SVM** (sử dụng liblinear / sklearn LinearSVC hoặc SGDClassifier) cho feature high-dim (VLAD/FV).
   * Nếu dùng histogram nhỏ, thử **chi2-kernel SVM** hoặc RBF nhưng tốn thời gian.
   * Tune C bằng grid search (ví dụ C ∈ {0.01, 0.1, 1, 10}).

8. **Đánh giá**

   * Dùng train/validation split (ví dụ train/val 80/20) hoặc k-fold. Chạy cross-validation để chọn K, C, kiểu encoding.
   * Báo accuracy, confusion matrix, per-class accuracy.

# 2) Các tham số gợi ý (giá trị khởi điểm)

* Upscale: 32→64 (bilinear).
* Dense SIFT: step=4, patch_size=8 or 12.
* PCA: giảm về 64 chiều.
* KMeans: K = 256, 512 (thử).
* VLAD clusters: K = 64–128 nếu dùng CNN features.
* SVM: linear, C = 1 (tune).
* SPM: levels = {1,2} (1×1 + 2×2 → 5 vùng).

# 3) Kỹ thuật quan trọng & lý do

* **Dense sampling > keypoints**: ảnh nhỏ không đủ keypoints ổn định; dense cho coverage đồng đều.
* **Upsampling trước extract**: giúp descriptor ổn định; nhưng nhớ không kỳ vọng “thông tin mới”.
* **PCA trước KMeans**: giảm chiều và giảm noise giúp clustering tốt hơn & nhanh hơn.
* **VLAD/FV > histogram**: chúng ghi lại residuals (thông tin hữu ích) nên mạnh hơn biểu đồ đơn thuần.
* **Power + L2 normalisation**: chống dominance của một số chiều, cải thiện SVM.
* **CNN features**: descriptors học được (semantic) thường tốt hơn handcrafted trên ảnh nhỏ.

# 4) Data augmentation & regularization

* Tăng dữ liệu train: random crop, horizontal flip, small rotations, color jitter — áp dụng khi trích descriptor hoặc khi training classifier.
* Khi dùng BoVW, nếu augment, **nên thêm các phiên bản augmented vào bước build vocabulary** để codebook phản ánh biến thể.

# 5) Thử nghiệm (ablation plan) — thứ tự ưu tiên

1. Baseline: dense SIFT (upsampled 64×64) + k=256 + BoW histogram + L2 + SVM linear → record acc.
2. Thêm SPM (1+2) → xem cải thiện.
3. Thử VLAD (vocab 64–128) thay cho BoW → thường nhảy lớn.
4. Chuyển sang CNN-descriptors (ResNet conv features) + VLAD + Linear SVM → so sánh. (Ưu tiên hàng đầu)
5. Thử Fisher Vectors nếu bạn có tool (VLFeat) — thường cải thiện hơn nữa.
6. Chọn params cuối cùng bằng grid search: K, PCA dim, SVM C.

# 6) Vấn đề hay gặp & cách khắc phục

* **Quá nhiều descriptors → KMeans chậm**: dùng MiniBatchKMeans, giảm PCA dim, hoặc lấy sample descriptors.
* **Overfit với vocab lớn**: giảm K hoặc tăng data augmentation.
* **VLAD vector quá lớn**: áp PCA lên VLAD + normalize.
* **SIFT license/implement**: OpenCV mới có SIFT (nonfree flag) — nếu không, dùng Dense RootSIFT (đổi histogram rồi L1 normalize sqrt) hoặc dùng CNN features.

# 7) Vì sao CNN-based approach thường tốt hơn

* CNN conv-features capture semantic patterns ở nhiều quy mô, tốt hơn cho ảnh có nhiễu & nhỏ.
* Quy trình: forward ảnh (32×32 or upsampled) → lấy feature map từ conv3/conv4 → flatten spatial vectors → KMeans → VLAD → PCA + normalize → Linear SVM.

# 8) Tài nguyên & công cụ khuyên dùng

* OpenCV (SIFT/ORB), VLFeat (VLAD/FV) hoặc sklearn + faiss (clustering), numpy, scikit-learn.
* PyTorch/TensorFlow để trích conv-features từ mô hình tiền huấn luyện.

# 9) Checklist nhanh để chạy ngay

* [ ] Chuẩn hoá ảnh, upscale → 64×64
* [ ] Trích descriptors: Dense SIFT (step=4, size=8) **OR** CNN conv vectors
* [ ] PCA descriptors → 64 dim
* [ ] MiniBatchKMeans (K=256) → codebook
* [ ] Encoding: VLAD (thử) hoặc histogram
* [ ] Power + L2 normalization
* [ ] Linear SVM (C=1), tune C
* [ ] Augmentation + SPM
* [ ] Eval on val-set, confusion matrix
