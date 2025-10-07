# BÀI TẬP LỚN - THẠC SĨ BÁCH KHOA HÀ NỘI
## Học phần: Thị giác máy tính
## Đề tài: Image Classification with BoVW method

---
## Nội dung đề tài
### Nội dung:
  - Mục đích là cài đạt phương pháp nhận dạng đối tựng sử dụng phương pháp túi từ (bag-of-words). 
  - Các kỹ thuật cụ thể cho từng giai đoạn (trích chọn đặc trưng, phân cụm và phân loại) được chọn bởi các thành viên trong nhóm.

### Dữ liệu thực nghiệm
- The CIFAR-10 dataset [data](https://www.cs.toronto.edu/~kriz/cifar.html)
- Dữ liệu bao gồm 60.000 ảnh màu với kích thước 32x32 cho 10 nhãn dữ liệu. Mỗi nhãn dữ liệu sẽ có 6000 ảnh. 
- Dữ liệu được chia thành 2 tập training với 50.000 ảnh và test với 10.000 ảnh.
- Link download: [link](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

Hướng dẫn đọc dữ liệu cifa:
- Download dữ liệu, unzip và đưa thư mục cifar-10-batches-py vào thư mục data 
- Thực hiện chạy lệnh sau để convert dữ liệu dạng byte về hình ảnh bản rõ (có thể nhìn bằng mắt) và tạo file train.csv và test.csv.
```bash
python data_process/read_cifar_metda.py
```

Sau khi chạy python script trên kết quả sẽ lưu trữ các ảnh vào thư mục `data/images` và tập train.csv và test.csv trong thư mục `dataset`.
Đây là định dạng cố định để sử dụng thực nghiệm trên bộ dữ liệu cifar.

---


