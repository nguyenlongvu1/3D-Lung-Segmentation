# Bắt đầu từ image Python 3.9
FROM python:3.9-slim

# Đặt thư mục làm việc, nơi code của bạn sẽ được mount vào
WORKDIR /code

# Sao chép CHỈ tệp requirements vào container
COPY ./requirements.txt /code/requirements.txt

# Cài đặt các thư viện
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Mở cổng 8000
EXPOSE 8000

# Lệnh để chạy ứng dụng của bạn
# Tệp code của bạn sẽ được HF tự động mount vào /code
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]