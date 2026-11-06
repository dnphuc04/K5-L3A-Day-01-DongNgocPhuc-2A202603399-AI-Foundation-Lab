# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603399
- **Họ và tên:** Đồng Ngọc Phúc
- **SEED:** 5313

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| `day01_lab.ipynb`| Đã có |
| [lab1_report.json](lab1_report.json) | Đã có; 15/15 kiểm tra đạt, đủ 10 dự đoán |
| [hand_calc.jpg](hand_calc.jpg) | Đã có tệp ảnh tính tay |

## 2. Kết quả chính

Em dùng kết quả lưu trong `day01_lab.ipynb`, đã đối chiếu khớp `lab1_report.json`. Sau bỏ trùng có 5.169 tin (4.516 ham, 653 spam); train/validation/test lần lượt có 3.619/775/775 tin, SUB gồm 20 spam và 20 ham. Bảng dưới dùng output của lần chạy đã lưu, không trộn thời gian từ các lần trước.

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0,903 | 0,947 | 0,150 | 0 |
| BoW + MLP | 0,920 | 0,974 | 0,781 | 0 |
| Encoder + kNN | 0,927 | 1,000 | 290,362 | 0 |
| Encoder + LogReg | 0,938 | 0,947 | 43,647 | 0 |
| Encoder + MLP head | 0,934 | 0,947 | 38,081 | 0 |
| LLM nhỏ zero-shot | — | 0,000 | 1919,187 | 0,007750 |
| LLM lớn zero-shot | — | 0,000 | 1364,652 | 0,077500 |
| LLM nhỏ few-shot | — | 0,000 | 1243,838 | 0,017083 |

LLM chỉ được đánh giá trên SUB. Các output đã quan sát có `invalid=40`, khiến toàn bộ nhãn không hợp lệ bị quy về ham; F1 bằng 0 chưa phản ánh chất lượng thực của LLM. Kiểm tra API giả lập đạt không chứng minh API thật trả nhãn đúng. Chi phí là ước tính theo bảng giá cấu hình, chưa phải số tiền tính phí được xác minh. Các phương án chạy cục bộ có chi phí API bằng 0 nhưng vẫn cần tài nguyên tính toán.

Trên SUB, đường Pareto gồm BoW + LogReg, BoW + MLP và Encoder + kNN. Trong các phương án đạt p95 dưới 50 ms, BoW + MLP có F1 SUB cao nhất (0,974), nhưng chênh lệch nhỏ trên 40 tin chưa đủ để khẳng định ưu thế chắc chắn. Trên toàn bộ test, Encoder + LogReg có F1 cao nhất (0,938).

## 3. Dự đoán và thực tế

1. **Stress test:** Em dự đoán Encoder + kNN đạt accuracy khoảng 65%, thực tế đạt 74%, cao hơn 9 điểm phần trăm. Các hàng xóm theo ngữ nghĩa hỗ trợ tốt hơn em dự đoán, nhất là ở nhóm thao túng và né từ khóa.
2. **F1 của BoW + LogReg:** Em dự đoán 0,83, thực tế đạt khoảng 0,903, cao hơn 0,073. Các đặc trưng từ vựng đủ hữu ích trên tập test cùng phân phối với train; tuy nhiên, F1 stress test chỉ 0,368 nên em không suy rộng kết quả này cho mọi loại spam.
3. **Tin ngụy trang:** Em dự đoán Encoder + LogReg đúng nhiều hơn BoW, nhưng cả hai cùng đúng 5/6. Cả hai bỏ lọt tin thúc giục xác nhận phần thưởng tài khoản, cho thấy embedding cũng có thể nhầm spam với thông báo hợp lệ.

## 4. Phân tích 50 case khó

| Phương án | F1 SUB thường | F1 50 case khó | Accuracy 50 case khó |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0,947 | 0,368 | 52% |
| BoW + MLP | 0,974 | 0,378 | 54% |
| Encoder + kNN | 1,000 | 0,764 | 74% |
| Encoder + LogReg | 0,947 | 0,667 | 66% |
| Encoder + MLP head | 0,947 | 0,667 | 66% |
| LLM nhỏ zero-shot | 0,000 | 0,000 | 42% |
| LLM nhỏ + prompt phòng thủ | 0,000 | 0,000 | 42% |

BoW + MLP tụt nhiều nhất: F1 từ 0,974 trên SUB xuống 0,378 trên 50 case khó, giảm 0,596. Accuracy chỉ 54%; nhóm né từ khóa đạt 0%, spam giọng tự nhiên đạt 10%. Test thường cùng phân phối với train và có thể chứa mẫu gần giống nhau nên chưa thể hiện hết điểm yếu này.

- **Case #12:** Tin mở đầu lịch sự, nói được bạn giới thiệu rồi dẫn tới trang hẹn hò. Em thấy BoW thiếu từ khóa mạnh, còn embedding có thể đặt tin gần hội thoại làm quen hợp lệ. Cả năm mô hình ngoài LLM đều bỏ lọt.
- **Case #13:** Tin giả nhắc ảnh sự kiện đã sẵn sàng, kèm liên kết đặt mua ảnh. Cách diễn đạt giống thông báo dịch vụ hợp lệ, khiến mô hình khó phân biệt quảng cáo không được yêu cầu. Cả năm mô hình ngoài LLM cũng bỏ lọt.

Encoder + kNN bền nhất trong các mô hình có kết quả phân loại hợp lệ: F1 0,764 và accuracy 74%. Hai hàng LLM cùng đoán toàn bộ là ham; em chưa thể kết luận về khả năng chống prompt injection khi chưa xác minh output gốc và lỗi đọc nhãn. Bộ 50 case là stress test tìm điểm yếu, không phải mẫu đại diện để ước tính hiệu quả ngoài đời.

**Bonus:** Em đã thêm 5 case có nhãn, nhóm và `why_tricky`; notebook in tổng 55 case. Tuy nhiên, output đánh giá và đọc lỗi vẫn là bộ 50 case gốc, chưa có kết quả đánh giá riêng 5 case mới. Vì vậy em chưa khẳng định case bonus nào đánh lừa mô hình. Các case được AI hỗ trợ đề xuất, đã khai báo ở AI log.

## 5. Thử thách cá nhân

**Đề #4:** Chạy kNN với k = 1, 5, 25, báo F1 và giải thích ảnh hưởng của k.

Em giữ nguyên embedding và tập train/test, chỉ thay đổi k; dùng `knn_predict` và `metrics` tự triển khai, không dùng sklearn.

| k | Precision | Recall | F1 test |
| ---: | ---: | ---: | ---: |
| 1 | 0,9091 | 0,9346 | 0,9217 |
| 5 | 0,9027 | 0,9533 | 0,9273 |
| 25 | 0,8707 | 0,9439 | 0,9058 |

Em thấy k=5 có F1 và recall cao nhất trong ba cấu hình, nhưng F1 chỉ nhỉnh hơn k=1 khoảng 0,0056 nên chưa khẳng định vượt trội. k quá nhỏ dễ bị hàng xóm nhiễu chi phối; k quá lớn lấy thêm tin ít liên quan. Thực tế k=25 giảm cả precision và recall so với k=5. Khi triển khai, em sẽ chọn k bằng validation rồi dùng test để đánh giá cuối.

## 6. Phương án cho sản phẩm thật

Em chọn **BoW + MLP làm phương án ban đầu** cho nhu cầu 10 triệu SMS/ngày: F1 test 0,920, p95 0,781 ms, thông lượng đo được khoảng 31.279 tin/giây và thời gian train 9,497 giây. Các số đo đáp ứng nhu cầu trung bình khoảng 116 tin/giây trong môi trường thử nghiệm; vẫn cần kiểm tra tải thực tế và lưu lượng cao điểm.

BoW + LogReg, BoW + MLP, Encoder + LogReg và Encoder + MLP đều đạt p95 dưới 50 ms. Em chấp nhận F1 thấp hơn Encoder + LogReg (0,938) để có tốc độ cao hơn; Encoder + LogReg chỉ đạt khoảng 21,836 tin/giây trong phép đo cả lô nên cần tăng năng lực xử lý. Encoder + kNN không đạt SLA với p95 290,362 ms.

Tuy nhiên, BoW + MLP chỉ đạt accuracy 54% trên stress test nên chưa đủ cơ sở triển khai ngay. Em đề xuất router: BoW + MLP xử lý tin chắc chắn, tin có xác suất trong [0,3; 0,7] chuyển sang Encoder + LogReg. Cần đánh giá lại F1, thông lượng và p95 toàn hệ thống vì router không bảo đảm cải thiện nếu chưa kiểm thử.

Với 300 triệu SMS trong tháng 30 ngày, chi phí theo số đo trong JSON ước tính là 2.325 USD cho model nhỏ zero-shot, 23.250 USD cho model lớn và 5.125 USD cho few-shot. Đây chỉ là ngoại suy từ cấu hình và output đang lỗi; cần xác minh lại sau khi xử lý lỗi LLM.

Điều em dự đoán sai là embedding sẽ tốt hơn BoW trên nhóm tin ngụy trang, nhưng cả hai cùng đúng 5/6. Em hiểu rằng mô hình phức tạp hơn không luôn thắng và cần đánh giá theo dữ liệu thực tế.

## 7. AI log và điều đã học

| Phần | Em hỏi AI gì | AI trả lời/viết gì | Em sửa hoặc kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 0 | Cách đọc notebook, chia dữ liệu, data leakage | Giải thích khái niệm và các cell | Đối chiếu 5.169 tin sau bỏ trùng, gồm 4.516 ham và 653 spam |
| Phần 1 | Ban đầu hỏi khái niệm; sau khi tự làm và được mentor đồng ý, hỏi code đối chiếu | Giải thích BoW, sigmoid, gradient, metrics và cung cấp code NumPy tham khảo | Em chạy kiểm tra, đối chiếu loss, precision, recall và F1; không dùng sklearn |
| Phần 2 | Ban đầu hỏi MLP, backpropagation, overfitting; sau khi tự làm và được mentor đồng ý, hỏi code đối chiếu | Giải thích và cung cấp code PyTorch tham khảo | Em phân tích biểu đồ train/val loss và kết quả test |
| Phần 3 | Ban đầu hỏi embedding, cosine, kNN; sau khi tự làm và được mentor đồng ý, hỏi code đối chiếu | Giải thích và cung cấp code NumPy tham khảo | Em chạy kiểm tra, đối chiếu kết quả 5/6 và biểu đồ PCA |
| Phần 4 | Cấu hình API, hoàn thành TODO, lỗi 429 và output không hợp lệ | Viết code gọi API, prompt, parse, batch; hướng dẫn giãn cách và retry | Em chạy zero-shot/few-shot, phát hiện invalid=40; nguyên nhân chưa được xác minh |
| Phần 5 | Phân tích Pareto, stress test, tính tay, phản ánh và báo cáo | Hỗ trợ diễn giải số liệu, cung cấp phép tính và đề xuất 5 case bonus | Em gửi output để đối chiếu; case bonus do AI đề xuất, không phải hoàn toàn tự nghĩ |
| Thử thách cá nhân | Sau khi tự làm và được mentor đồng ý, nhờ đối chiếu kNN với k=1, 5, 25 | Cung cấp code tham khảo và giải thích ảnh hưởng của k | Em chạy và đối chiếu F1 lần lượt 0,9217; 0,9273; 0,9058 |

Em hiểu rằng điểm test cao chưa bảo đảm mô hình xử lý tốt các trường hợp khó và mô hình lớn hơn không luôn tốt hơn. Em phân biệt được kiểm tra giả lập đạt với việc API thật trả nhãn hợp lệ. Khi chọn phương án, em cần cân nhắc đồng thời chất lượng, độ trễ, thông lượng, chi phí và độ bền trước dữ liệu mới.
