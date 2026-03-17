from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import threading

from cat_dog import ImagePredictor, TrainingConfig
from setup_gpu import TensorFlowConfig


class PredictionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Dự đoán Chó vs Mèo")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        # Configure style
        self.root.configure(bg="#f0f0f0")
        self.style_setup()

        # Initialize GPU and model
        TensorFlowConfig.init_gpu()
        config = TrainingConfig()

        try:
            self.predictor = ImagePredictor(
                model_path=config.best_model_path,
                image_size=config.image_size,
            )
        except FileNotFoundError as e:
            messagebox.showerror("Lỗi", f"Không thể tải model: {e}")
            self.root.destroy()
            return

        self.current_image_path: Path | None = None
        self.is_predicting = False

        self.setup_ui()

    def style_setup(self) -> None:
        """Configure colors and fonts"""
        self.bg_color = "#f0f0f0"
        self.card_color = "#ffffff"
        self.primary_color = "#2196F3"
        self.success_color = "#4CAF50"
        self.text_color = "#333333"

    def setup_ui(self) -> None:
        """Setup all UI components"""
        # Header
        header_frame = tk.Frame(self.root, bg=self.primary_color, height=80)
        header_frame.pack(fill=tk.X)

        header_label = tk.Label(
            header_frame,
            text="Nhận diện Chó vs Mèo",
            font=("Arial", 24, "bold"),
            bg=self.primary_color,
            fg="white"
        )
        header_label.pack(pady=15)

        # Main content frame
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left side - Image display
        left_frame = tk.Frame(content_frame, bg=self.card_color, relief=tk.FLAT, bd=0)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self._add_shadow_effect(left_frame)

        image_label = tk.Label(
            left_frame,
            text="Chọn ảnh để bắt đầu",
            font=("Arial", 14),
            bg=self.card_color,
            fg="#999999",
            height=12
        )
        image_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.image_label = image_label

        # Button frame below image
        button_frame = tk.Frame(content_frame, bg=self.card_color, relief=tk.FLAT, bd=0)
        button_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self._add_shadow_effect(button_frame)

        # Select button
        select_btn = tk.Button(
            button_frame,
            text="Chọn Ảnh",
            font=("Arial", 12, "bold"),
            bg=self.primary_color,
            fg="white",
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            command=self.select_image,
            padx=20,
            pady=15
        )
        select_btn.pack(pady=(15, 10), padx=15, fill=tk.X)

        # Predict button
        self.predict_btn = tk.Button(
            button_frame,
            text="Dự đoán",
            font=("Arial", 12, "bold"),
            bg=self.success_color,
            fg="white",
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            command=self.run_prediction,
            padx=20,
            pady=15,
            state=tk.DISABLED
        )
        self.predict_btn.pack(pady=10, padx=15, fill=tk.X)

        # Result frame
        result_frame = tk.Frame(button_frame, bg="#f9f9f9", relief=tk.FLAT, bd=0)
        result_frame.pack(pady=(20, 15), padx=15, fill=tk.BOTH, expand=True)

        result_title = tk.Label(
            result_frame,
            text="Kết quả dự đoán:",
            font=("Arial", 11, "bold"),
            bg="#f9f9f9",
            fg=self.text_color
        )
        result_title.pack(anchor=tk.W, pady=(0, 10))

        # Result display
        self.result_text = tk.Text(
            result_frame,
            font=("Arial", 10),
            bg="white",
            fg=self.text_color,
            relief=tk.FLAT,
            bd=1,
            height=8,
            width=30,
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="Xóa",
            font=("Arial", 11),
            bg="#f44336",
            fg="white",
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            command=self.clear_all,
            padx=20,
            pady=10
        )
        clear_btn.pack(pady=(0, 15), padx=15, fill=tk.X)

    def _add_shadow_effect(self, frame: tk.Frame) -> None:
        """Add a shadow effect to a frame"""
        frame.configure(relief=tk.RAISED, bd=2)

    def select_image(self) -> None:
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = Path(file_path)
            self.display_image(self.current_image_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.update_result("")

    def display_image(self, image_path: Path) -> None:
        """Display selected image"""
        try:
            image = Image.open(image_path)
            # Resize to fit in label
            image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            self.image_label.config(image=photo, text="")
            self.image_label.photo = photo
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở ảnh: {e}")

    def run_prediction(self) -> None:
        """Run prediction in a separate thread"""
        if not self.current_image_path or self.is_predicting:
            return

        self.is_predicting = True
        self.predict_btn.config(state=tk.DISABLED, text="⏳ Đang xử lý...")

        thread = threading.Thread(target=self._predict_worker)
        thread.daemon = True
        thread.start()

    def _predict_worker(self) -> None:
        """Worker thread for prediction"""
        try:
            result = self.predictor.predict(self.current_image_path)

            result_text = (
                f"NHÃN DỰ ĐOÁN: {result.label.upper()}\n\n"
                f"Độ tin cậy: {result.confidence * 100:.1f}%\n\n"
                f"Xác suất là Chó: {result.dog_probability * 100:.1f}%\n"
                f"Xác suất là Mèo: {(1 - result.dog_probability) * 100:.1f}%"
            )

            self.root.after(0, lambda: self.update_result(result_text))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi dự đoán: {e}"))
        finally:
            self.is_predicting = False
            self.root.after(0, lambda: self.predict_btn.config(
                state=tk.NORMAL,
                text="Dự đoán"
            ))

    def update_result(self, text: str) -> None:
        """Update result display"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state=tk.DISABLED)

    def clear_all(self) -> None:
        """Clear all selections and results"""
        self.current_image_path = None
        self.image_label.config(image="", text="Chọn ảnh để bắt đầu")
        self.image_label.photo = None
        self.update_result("")
        self.predict_btn.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
