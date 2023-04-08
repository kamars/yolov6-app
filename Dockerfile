# Dựa trên image cơ bản nào
FROM python:3.8.16-slim-bullseye

RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean

# Khai báo thư mục làm việc
WORKDIR /app

# Copy toàn bộ file mã nguồn và các file khác vào image
COPY . .

# Cài đặt thư viện

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install -r requirements.txt

# Thực hiện lệnh chạy
CMD ["python","./my_api.py"]