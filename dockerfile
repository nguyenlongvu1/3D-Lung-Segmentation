# Bắt đầu từ image Python 3.9
FROM python:3.9-slim

# Đặt thư mục làm việc bên trong container
WORKDIR /code

# Sao chép tệp requirements vào container
COPY ./requirements.txt /code/requirements.txt

# Cài đặt các thư viện
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Sao chép toàn bộ code của bạn vào container
COPY ./app /code/app
COPY ./static /code/static
COPY ./models /code/models

# Mở cổng 8000 (phải khớp với app_port trong README.md)
EXPOSE 8000

# Lệnh để chạy ứng dụng FastAPI của bạn
# Nó chạy uvicorn, trỏ đến tệp app.main (app/main.py) và đối tượng 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]