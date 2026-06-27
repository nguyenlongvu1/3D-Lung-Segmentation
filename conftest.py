# File rỗng có chủ đích: pytest thấy conftest.py ở gốc repo sẽ thêm thư mục này vào
# sys.path, nhờ đó `pytest` (console script) import được `ml`/`api` mà không cần
# `python -m pytest`. Quan trọng cho CI.
