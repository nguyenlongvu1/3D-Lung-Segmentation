# Bắt đầu từ image Python 3.9
FROM python:3.9-slim

# Đặt thư mục làm việc, nơi code của bạn sẽ được mount vào
WORKDIR /code

# Sao chép CHỈ tệp requirements vào container
# Làm điều này trước để tận dụng cache của Docker
COPY ./requirements.txt /code/requirements.txt

# Cài đặt các thư viện
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- LỖI SỐ 1 ĐƯỢC SỬA Ở ĐÂY ---
# Ghi chú "HF tự động mount" của bạn là KHÔNG ĐÚNG với 'sdk: docker'.
# Bạn PHẢI tự sao chép code của mình (thư mục app, static, v.v.)
# Lệnh "COPY . ." sẽ sao chép mọi thứ vào thư mục /code.
COPY . .

# Lệnh để chạy ứng dụng của bạn
EXPOSE 7860

# --- LỖI SỐ 2 ĐƯỢC SỬA Ở ĐÂY ---
# Lệnh đúng phải là "app.main:app", không phải "main:app"
# Vì file main.py của bạn nằm trong thư mục 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
