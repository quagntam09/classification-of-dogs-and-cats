import os
import tensorflow as tf

class TensorFlowConfig:
    @staticmethod
    def init_gpu():
        """
        Cấu hình tối ưu cho TensorFlow chạy trên GPU (WSL2/Linux).
        Bao gồm: Ẩn log dư thừa, tắt oneDNN và bật Memory Growth.
        """
        # 1. Thiết lập biến môi trường (phải chạy trước khi import tf nặng)
        # Ẩn log INFO và WARNING (chỉ hiện ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Tắt thông báo oneDNN để đảm bảo độ chính xác số học ổn định
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        # 2. Cấu hình Memory Growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"--- Đã kích hoạt GPU: {len(gpus)} thiết bị ---")
                print(f"--- Chế độ: Memory Growth đã được bật ---")
            except RuntimeError as e:
                # Lỗi này thường xảy ra nếu GPU đã được khởi tạo trước khi gọi hàm này
                print(f"Lỗi cấu hình GPU: {e}")
        else:
            print("Cảnh báo: Không tìm thấy GPU. Hệ thống sẽ chạy bằng CPU.")

# Bạn có thể để dòng này để khi import nó tự chạy luôn (tùy chọn)
TensorFlowConfig.init_gpu()