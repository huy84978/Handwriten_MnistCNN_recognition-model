Giới thiệu các công cụ sử dụng
Ngôn ngữ lập trình Python: Python là một ngôn ngữ lập trình cấp cao, đa năng và được biết đến với cú pháp đơn giản, dễ đọc, phù hợp cho cả người mới học lẫn các chuyên gia lập trình. Với khả năng hỗ trợ nhiều thư viện mạnh mẽ, Python trở thành lựa chọn hàng đầu trong lĩnh vực trí tuệ nhân tạo, học máy, xử lý dữ liệu, và phát triển ứng dụng.
Thư viện TensorFlow và Keras: Được sử dụng để xây dựng và nạp mô hình mạng neural tích chập (CNN) đã được huấn luyện trước đó . Keras cung cấp một giao diện lập trình cấp cao để xây dựng mô hình CNN dễ dàng hơn. 
NumPy: Thư viện tính toán khoa học được sử dụng để thao tác với các mảng dữ liệu, chẳng hạn như hình ảnh được đưa vào mô hình . 
Tkinter: Thư viện đồ họa của Python được sử dụng để tạo giao diện người dùng (GUI) cho ứng dụng . Tkinter cung cấp các widget như canvas, nút và nhãn để người dùng tương tác với chương trình. 
Pillow (PIL Fork): Thư viện xử lý hình ảnh của Python được sử dụng để thao tác với ảnh do người dùng vẽ trên canvas. PIL cung cấp các hàm để tạo, mở, lưu và xử lý ảnh.


mô hình giúp cho người dùng nhận dạnh hình ảnh chữ số viết tay và thực hiện nhận dạng chính xác chữ số từ 0 đến 9. Kết quả dự đoán cần được hiển thị rõ ràng và dự đoán chữ số một cách tốt để có thể dùng trên giao diện người dùng xây dựng. mô hình cần đảm bảo thời gian xử lý nhanh, đáp ứng nhu cầu sử dụng thực tế(bước đầu là cho phép người dùng làm quen với học sâu của xử lý ảnh).

Lựa chọn bộ dữ liệu 
Bộ dữ liệu được sử dụng trong đề tài là: MNIST 
Bộ dữ liệu MNIST (Modified National Institute of Standards and Technology) là một tập dữ liệu chuẩn trong lĩnh vực trí tuệ nhân tạo, đặc biệt trong bài toán nhận dạng chữ số viết tay. MNIST bao gồm 70.000 hình ảnh chữ số viết tay từ 0 đến 9, trong đó có 60.000 hình ảnh dùng để huấn luyện và 10.000 hình ảnh để kiểm tra. Mỗi hình ảnh trong bộ dữ liệu có kích thước 28x28 pixel và là ảnh xám (grayscale), giúp giảm độ phức tạp của bài toán và tăng tính đồng nhất của dữ liệu . Đặc biệt, các chữ số trong MNIST được viết bởi nhiều người khác nhau, bao gồm cả trẻ em và người lớn, tạo nên sự đa dạng trong phong cách viết và độ phức tạp, làm cho nó trở thành một bộ dữ liệu lý tưởng để kiểm tra độ mạnh mẽ của các mô hình nhận diện.
Trong nhận dạng chữ viết tay, MNIST đóng vai trò là bộ dữ liệu khởi điểm để phát triển và kiểm thử các thuật toán xử lý ảnh và học sâu . Bộ dữ liệu này thường được sử dụng để huấn luyện các mô hình Machine Learning hoặc Deep Learning, đặc biệt là mạng nơ-ron tích chập (Convolutional Neural Networks - CNN). Nhờ tính chuẩn hóa cao và dễ sử dụng, MNIST không chỉ giúp các nhà nghiên cứu kiểm chứng hiệu quả của mô hình mà còn hỗ trợ trong việc so sánh hiệu suất giữa các thuật toán . Ngoài ra, MNIST còn được dùng làm tài liệu học tập cho người mới bắt đầu nghiên cứu về trí tuệ nhân tạo, giúp họ hiểu rõ hơn về quy trình xử lý dữ liệu, huấn luyện mô hình và đánh giá kết quả.
Ứng dụng thực tiễn của MNIST chủ yếu nằm trong việc xây dựng nền tảng cho các hệ thống nhận diện chữ viết tay, chẳng hạn như xử lý tài liệu hành chính, nhận diện séc hoặc hóa đơn trong ngân hàng, và số hóa văn bản viết tay . Bộ dữ liệu này không chỉ hỗ trợ việc phát triển các hệ thống nhỏ mà còn là bước khởi đầu cho các nghiên cứu mở rộng với dữ liệu lớn và phức tạp hơn, chẳng hạn như nhận diện chữ viết tay toàn phần hoặc các ký tự không phải tiếng Anh.
Bộ dữ liệu MNIST chứa hình ảnh thang độ xám của chữ số viết tay, cung cấp một bộ dữ liệu có cấu trúc tốt cho các tác vụ phân loại hình ảnh. Dưới đây là một ví dụ về hình ảnh từ bộ dữ liệu:

**Hình 1:** Các mẫu chữ số viết tay từ tập dữ liệu MNIST.  
![Các mẫu chữ số viết tay từ tập dữ liệu MNIST](mnist_samples.png)

### Mô tả hình ảnh:
Hình ảnh là các chữ số viết tay từ bộ dữ liệu MNIST, với mỗi ô vuông đại diện cho một mẫu chữ số. Có tổng cộng 10 hàng, mỗi hàng tương ứng với một chữ số từ 0 đến 9. Mỗi chữ số được viết theo nhiều cách khác nhau, thể hiện sự đa dạng về kiểu dáng, độ nghiêng, và nét viết tay.

### Chú thích chi tiết:
- **Hàng 1:** Chữ số 0, hiển thị nhiều kiểu viết tay khác nhau, từ nét tròn đến nét góc cạnh.
- **Hàng 2:** Chữ số 1, với các nét thẳng đứng, nghiêng hoặc hơi cong.
- **Hàng 3:** Chữ số 2, thể hiện các cách viết từ nét mượt đến nét góc cạnh.
- **Hàng 4:** Chữ số 3, với các vòng tròn trên và dưới được viết ở nhiều dạng.
- **Hàng 5:** Chữ số 4, có kiểu nét đứng thẳng hoặc nghiêng.
- **Hàng 6:** Chữ số 5, với nét gấp khúc hoặc đường cong mềm mại.
- **Hàng 7:** Chữ số 6, hiển thị các dạng cong tròn rõ ràng.
- **Hàng 8:** Chữ số 7, với nét ngang trên cùng và nét nghiêng đi xuống.
- **Hàng 9:** Chữ số 8, hiển thị các dạng viết tay với hai vòng tròn đều nhau hoặc không đều.
- **Hàng 10:** Chữ số 9, có nét trên tròn và nét dưới thẳng hoặc cong.
