1. Lý do ORB không hoạt động tốt với CIFAR

| Nguyên nhân                             | Giải thích                                                                                                                                              |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ảnh CIFAR quá nhỏ (32×32 px)**        | ORB cần vùng đủ lớn để phát hiện *corner* (góc, điểm đặc trưng). Với 32×32 thì vùng lân cận rất hạn chế, gần như không đủ thông tin để tạo *keypoints*. |
| **Độ phân giải thấp, nhiều nhiễu**      | Ảnh CIFAR có chi tiết mờ, tạp, và scale rất nhỏ – các thuật toán dựa trên gradient hoặc corner như FAST/SIFT/SURF/ORB sẽ gặp khó.                       |
| **Ảnh màu RGB, không rõ cấu trúc cạnh** | ORB thường hoạt động tốt trên ảnh xám với cấu trúc rõ ràng (cạnh, biên, texture rõ rệt). CIFAR nhiều ảnh phẳng, vùng màu đồng nhất.                     |
| **Tính bất biến theo scale bị hạn chế** | ORB không mạnh với thay đổi tỉ lệ, mà CIFAR thường có vật thể chiếm tỷ lệ nhỏ và thay đổi theo ảnh.                                                     |


2. Điều chỉnh tham số

| Tham số           | Mặc định               | Giải thích                             | Ảnh hưởng với CIFAR                                                                      |
| ----------------- | ---------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------- |
| **nfeatures**     | 500                    | Số lượng feature tối đa muốn detect    | Có thể tăng lên để cố gắng lấy thêm điểm                                                 |
| **scaleFactor**   | 1.2                    | Hệ số scale giữa các pyramid levels    | Nếu ảnh nhỏ, nên giảm xuống (1.05–1.1) để có nhiều level chi tiết hơn                    |
| **nlevels**       | 8                      | Số tầng pyramid scale                  | Ảnh 32×32 mà 8 tầng thì tầng cuối gần như 0 pixel 😅 → nên giảm còn 3–4                  |
| **edgeThreshold** | 31                     | Kích thước vùng bỏ qua biên khi detect | Rất lớn so với ảnh nhỏ, nghĩa là gần như bỏ hết vùng có thể detect → nên giảm mạnh (3–5) |
| **patchSize**     | 31                     | Kích thước patch mô tả                 | Ảnh 32×32 mà patch 31 thì chỉ còn 1 pixel biên 😬 → nên giảm còn 15 hoặc 9               |
| **fastThreshold** | 20                     | Ngưỡng FAST (độ nhạy corner)           | Giảm xuống (5–10) để phát hiện nhiều điểm hơn trong ảnh mờ                               |
| **scoreType**     | `cv2.ORB_HARRIS_SCORE` | Dùng Harris để chọn góc (ổn)           | Giữ nguyên cũng được                                                                     |
