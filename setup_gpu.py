import os
import shutil
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

LOCAL_CUDA_COMPAT_DIR = Path(__file__).resolve().parent / ".cuda_compat"
LOCAL_LIBDEVICE = LOCAL_CUDA_COMPAT_DIR / "nvvm" / "libdevice" / "libdevice.10.bc"

if Path("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc").is_file():
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda")
elif Path("/opt/cuda/nvvm/libdevice/libdevice.10.bc").is_file():
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/opt/cuda")
elif Path("/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc").is_file():
    LOCAL_LIBDEVICE.parent.mkdir(parents=True, exist_ok=True)
    if not LOCAL_LIBDEVICE.exists():
        toolkit_libdevice = Path("/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc")
        try:
            LOCAL_LIBDEVICE.symlink_to(toolkit_libdevice)
        except OSError:
            shutil.copy2(toolkit_libdevice, LOCAL_LIBDEVICE)
    os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={LOCAL_CUDA_COMPAT_DIR}")

import tensorflow as tf

class TensorFlowConfig:
    @staticmethod
    def _has_cuda_libdevice() -> bool:
        candidates = [
            Path("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"),
            Path("/opt/cuda/nvvm/libdevice/libdevice.10.bc"),
            LOCAL_LIBDEVICE,
            Path("/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc"),
            Path(tf.__file__).resolve().parent / ".." / "nvidia" / "cuda_nvcc" / "nvvm" / "libdevice" / "libdevice.10.bc",
        ]
        return any(path.resolve().is_file() for path in candidates)

    @staticmethod
    def init_gpu():
        """
        Cấu hình TensorFlow chạy GPU an toàn cho Linux/WSL2.
        Bao gồm: bật Memory Growth để tránh chiếm toàn bộ VRAM ngay từ đầu.
        """
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                if not TensorFlowConfig._has_cuda_libdevice():
                    tf.config.set_visible_devices([], "GPU")
                    print("Cảnh báo: Thiếu CUDA libdevice.10.bc, chuyển sang chạy CPU để tránh lỗi JIT.")
                    return

                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"--- Đã kích hoạt GPU: {len(gpus)} thiết bị ---")
                print(f"--- Chế độ: Memory Growth đã được bật ---")
            except RuntimeError as e:
                print(f"Lỗi cấu hình GPU: {e}")
        else:
            print("Cảnh báo: Không tìm thấy GPU. Hệ thống sẽ chạy bằng CPU.")