import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QComboBox, QFileDialog, QGraphicsView, QGraphicsScene, QWidget, QMessageBox, QScrollArea, QSizePolicy, QGraphicsRectItem, QGridLayout
)
from PyQt5.QtCore import Qt,QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # Image data
        self.image = None
        self.output1 = None
        self.output2 = None

        # Main Layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Buttons
        self.create_buttons()

        # Viewports
        self.create_viewports()

    
    def create_buttons(self):
        button_layout = QGridLayout()
        row, col = 0, 0
        
        def add_to_grid(widget):
            nonlocal row, col
            button_layout.addWidget(widget, row, col)
            row += 1
            if row >= 2:  # Change 5 to control the number of buttons per row
               row = 0
               col += 1

        # Open Image Button
        self.open_btn = QPushButton("Open Image")
        self.open_btn.clicked.connect(self.open_image)
        add_to_grid(self.open_btn)

        # Target Viewport Selector
        self.target_combo = QComboBox()
        self.target_combo.addItems(["Target: Input", "Target: Output1", "Target: Output2"])
        add_to_grid(self.target_combo)

        # Target Interpolation Selector
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["Linear", "Area", "Cubic", "Nearest"])
        add_to_grid(self.interpolation_combo)

        # Zoom
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["zoom: 1x", "zoom: 2x", "zoom: 4x", "zoom: 0.5x"])
        self.zoom_combo.currentIndexChanged.connect(self.zoom_image)
        add_to_grid(self.zoom_combo)

        # Brightness Slider
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-20)
        self.brightness_slider.setMaximum(20)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setToolTip("Adjust Brightness")
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        add_to_grid(self.brightness_slider)

        # Contrast Slider
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-20)
        self.contrast_slider.setMaximum(20)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setToolTip("Adjust Contrast")
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)
        add_to_grid(self.contrast_slider)

        # Noise
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["Add Noise: None", "Gaussian", "Salt & Pepper", "Speckle"])
        self.noise_combo.currentIndexChanged.connect(self.add_noise)
        add_to_grid(self.noise_combo)

        # Denoising
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["Denoise: None", "Median Filter", "Gaussian Filter", "Bilateral Filter"])
        self.denoise_combo.currentIndexChanged.connect(self.apply_denoise)
        add_to_grid(self.denoise_combo)

        # Contrast
        self.contrast_combo = QComboBox()
        self.contrast_combo.addItems(["Enhance Contrast: None", "Enhance Contrast: CLAHE", "Enhance Contrast: Histogram Equalization", "Enhance Contrast: Logarithmic"])
        self.contrast_combo.currentIndexChanged.connect(self.enhance_contrast)
        add_to_grid(self.contrast_combo)

        # Filters
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Filter: None", "High Pass", "Low Pass"])
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        add_to_grid(self.filter_combo)

        # SNR Button
        self.snr_btn = QPushButton("Calculate SNR")
        self.snr_btn.clicked.connect(self.open_snr_window)
        add_to_grid(self.snr_btn)

        # CNR Button
        self.cnr_btn = QPushButton("Calculate CNR", self)
        self.cnr_btn.clicked.connect(self.open_cnr_window)
        add_to_grid(self.cnr_btn)

        reset_button = QPushButton("Reset Viewports")
        reset_button.clicked.connect(self.reset_viewports)
        self.layout.addWidget(reset_button)

        self.active_image = None

        # Add grid layout to the main layout
        self.layout.addLayout(button_layout)

    def create_viewports(self):
        viewport_layout = QHBoxLayout()

        # Input Viewport
        self.input_view = QLabel("Input")
        self.input_view.setFixedSize(600, 600)
        self.input_view.setStyleSheet("border: 2px solid black;")
        self.input_view.mousePressEvent = lambda event: self.set_active_viewport(self.input_view)
        self.input_view.mouseDoubleClickEvent = lambda event: self.show_histogram(self.image, "Input Image Histogram")
        viewport_layout.addWidget(self.input_view)

        # Output Viewport 1
        self.output1_view = QLabel("Output 1")
        self.output1_view.setFixedSize(600, 600)
        self.output1_view.setStyleSheet("border: 1px solid black;")
        self.output1_view.mousePressEvent = lambda event: self.set_active_viewport(self.output1_view)
        self.output1_view.mouseDoubleClickEvent = lambda event: self.show_histogram(self.output1, "Output1 Image Histogram")
        viewport_layout.addWidget(self.output1_view)

        # Output Viewport 2
        self.output2_view = QLabel("output 2")
        self.output2_view.setFixedSize(600, 600)
        self.output2_view.setStyleSheet("border: 1px solid black;")
        self.output2_view.mousePressEvent = lambda event: self.set_active_viewport(self.output2_view)
        self.output2_view.mouseDoubleClickEvent = lambda event: self.show_histogram(self.output2, "Output2 Image Histogram")
        viewport_layout.addWidget(self.output2_view)

        # Set QLabel properties
        for view in [self.input_view, self.output1_view, self.output2_view]:
            view.setAlignment(Qt.AlignCenter)
            view.setScaledContents(False)  # Keep actual size; do not stretch
            view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # Allow dynamic resizing

        # Initialize active viewport
        self.active_viewport = self.input_view

        # Add layout to main
        self.layout.addLayout(viewport_layout)

        # Default active viewport
        self.set_active_viewport(self.input_view)

    def set_active_viewport(self, viewport):
        """Set the clicked viewport as active and highlight it."""
        # Reset styles for all viewports
        for view in [self.input_view, self.output1_view, self.output2_view]:
           view.setStyleSheet("border: 1px solid black;")

        # Highlight the active viewport
        viewport.setStyleSheet("border: 2px solid blue;")
        self.active_viewport = viewport 
    
    def reset_viewports(self):
        """Resets all viewport images to their default state."""
        # Reset input viewport
        self.input_view.clear()
        self.input_view.setText("Input")
        self.input_view.setStyleSheet("border: 2px solid black;")
    
        # Reset output viewport 1
        self.output1_view.clear()
        self.output1_view.setText("Output 1")
        self.output1_view.setStyleSheet("border: 1px solid black;")
    
        # Reset output viewport 2
        self.output2_view.clear()
        self.output2_view.setText("Output 2")
        self.output2_view.setStyleSheet("border: 1px solid black;")
    
        # Clear any active images
        self.image = None
        self.output1 = None
        self.output2 = None
        print("Viewports have been reset to their default state.")


    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.input_view, self.image)

    def display_image(self, label, image):
      if image is not None:
        try:
            print(f"Displaying image with shape: {image.shape}")
            height, width = image.shape[:2]

            if len(image.shape) == 2:  # Grayscale
                bytes_per_line = width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:  # Color (BGR to RGB)
                bytes_per_line = width * 3
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            pixmap = QPixmap.fromImage(q_image)

            # Set the pixmap directly without scaling
            label.setPixmap(pixmap)

        except Exception as e:
            print(f"Error displaying image: {e}")

    def open_cnr_window(self):

        if "Input" in self.target_combo.currentText():
            active_image = self.image
        elif "Output1" in self.target_combo.currentText():
            active_image = self.output1
        elif "Output2" in self.target_combo.currentText():
            active_image = self.output2
        else:
            return

        if active_image is None:
            QMessageBox.warning(self, "Error", "No active viewport image available!")
            return
        self.cnr_window = CalculationWindow(self, mode="CNR", image=active_image)
        self.cnr_window.show()

    def open_snr_window(self):

        if "Input" in self.target_combo.currentText():
            active_image = self.image
        elif "Output1" in self.target_combo.currentText():
            active_image = self.output1
        elif "Output2" in self.target_combo.currentText():
            active_image = self.output2
        else:
            return
        
        if active_image is None:
            QMessageBox.warning(self, "Error", "No active viewport image available!")
            return
        self.snr_window = CalculationWindow(self, mode="SNR", image=active_image)
        self.snr_window.show()

    def apply_filter(self):
        """Apply filters (high-pass or low-pass) to the selected target image."""
        # Get the target image
        if "Input" in self.target_combo.currentText():
            target_image = self.image
        elif "Output1" in self.target_combo.currentText():
            target_image = self.output1
        elif "Output2" in self.target_combo.currentText():
            target_image = self.output2
        else:
            return

        if target_image is not None:
            filter_type = self.filter_combo.currentText()
            filtered_image = target_image.copy()

            if "High Pass" in filter_type:
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                filtered_image = cv2.filter2D(target_image, -1, kernel)

            elif "Low Pass" in filter_type:
                kernel = np.ones((5, 5), np.float32) / 25
                filtered_image = cv2.filter2D(target_image, -1, kernel)

            self.update_viewport(self.active_viewport, filtered_image)

    def get_target_viewport(self):
        """Get the target viewport selected by the user."""
        target = self.target_combo.currentText()
        if "Input" in target:
            return self.input_view
        elif "Output1" in target:
            return self.output1_view
        elif "Output2" in target:
            return self.output2_view
        return None

    def zoom_image(self):
      if self.image is not None and self.active_viewport is not None:
        try:
            # Extract the zoom factor from the dropdown and convert to float
            zoom_factor = float(self.zoom_combo.currentText().split(":")[1][:-1].replace('x', ''))
            print(f"Zoom factor: {zoom_factor}")

            # Map dropdown menu selection to OpenCV interpolation types
            interpolation_map = {
                "Nearest": cv2.INTER_NEAREST,
                "Linear": cv2.INTER_LINEAR,
                "Cubic": cv2.INTER_CUBIC,
                "Area": cv2.INTER_AREA,
            }

            # Get the selected interpolation type
            selected_interpolation = self.interpolation_combo.currentText()
            print(f"Selected interpolation: {selected_interpolation}")
            interpolation_type = interpolation_map.get(selected_interpolation, cv2.INTER_LINEAR)

            # Resize the image using the zoom factor and interpolation type
            zoomed_image = cv2.resize(self.image, None, fx=zoom_factor, fy=zoom_factor, interpolation=interpolation_type)
            print(f"Zoomed image shape: {zoomed_image.shape}")

            # Display the zoomed image
            self.display_image(self.active_viewport, zoomed_image)
            self.active_viewport.resize(zoomed_image.shape[1], zoomed_image.shape[0])  

        except Exception as e:
            print(f"Error during zooming: {e}")

    def reset_zoom(self):
       if self.image is not None:
          self.display_image(self.input_view, self.image)

    def adjust_brightness(self):
        if self.active_viewport in [self.input_view, self.output1_view, self.output2_view]:
            image = self.image if self.active_viewport == self.input_view else \
                    self.output1 if self.active_viewport == self.output1_view else self.output2
        if image is not None:
            brightness = self.brightness_slider.value()
            bright_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness)
            self.update_viewport(self.active_viewport,bright_image)

    def adjust_contrast(self):
        if self.active_viewport in [self.input_view, self.output1_view, self.output2_view]:
            image = self.image if self.active_viewport == self.input_view else \
                    self.output1 if self.active_viewport == self.output1_view else self.output2
            if image is not None:
                contrast = self.contrast_slider.value()
                alpha = 1 + (contrast / 100.0)
                contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
                self.update_viewport(self.active_viewport,contrast_image)
    
   
    def display_image_on_view(self, label, image):
        """Display an image on the given label."""
        if image is not None:
           height, width = image.shape
           bytes_per_line = width
           q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
           pixmap = QPixmap.fromImage(q_image)
           label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))

    def add_noise(self):
        """Add noise to the selected viewport's image based on the dropdown selection."""
       
        if "Input" in self.target_combo.currentText():
            target_image = self.image
        elif "Output1" in self.target_combo.currentText():
            target_image = self.output1
        elif "Output2" in self.target_combo.currentText():
            target_image = self.output2
        else:
            return

        if target_image is not None:
            noise_type = self.noise_combo.currentText()
            noisy_image = target_image.copy()

            if "Gaussian" in noise_type:
                mean, sigma = 0, 25
                gaussian = np.random.normal(mean, sigma, target_image.shape).astype(np.uint8)
                noisy_image = cv2.add(target_image, gaussian)

            elif "Salt & Pepper" in noise_type:
                salt_pepper = np.copy(target_image)
                num_salt = np.ceil(0.02 * target_image.size)
                num_pepper = np.ceil(0.02 * target_image.size)
                coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in target_image.shape]
                coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in target_image.shape]
                salt_pepper[coords_salt[0], coords_salt[1]] = 255
                salt_pepper[coords_pepper[0], coords_pepper[1]] = 0
                noisy_image = salt_pepper

            elif "Speckle" in noise_type:
                speckle = np.random.randn(*target_image.shape) * 0.2
                noisy_image = target_image + target_image * speckle
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

            self.update_viewport(self.active_viewport, noisy_image)

    def apply_denoise(self):
            
            if "Input" in self.target_combo.currentText():
                target_image = self.image
            elif "Output1" in self.target_combo.currentText():
                target_image = self.output1
            elif "Output2" in self.target_combo.currentText():
                target_image = self.output2
            else:
                 return
            
            if target_image is not None:
                denoise_type = self.denoise_combo.currentText()
                denoised_image = target_image.copy()

                if "Median Filter" in denoise_type:
                     denoised_image = cv2.medianBlur(self.image, 5)

                elif "Gaussian Filter" in denoise_type:
                      denoised_image = cv2.GaussianBlur(self.image, (5, 5), 0)

                elif "Bilateral Filter" in denoise_type:
                      denoised_image = cv2.bilateralFilter(self.image, 9, 75, 75)

                self.update_viewport(self.active_viewport, denoised_image)
    
    def enhance_contrast(self):
        """Enhance the contrast of the selected viewport's image based on the dropdown selection."""
    
        # Determine the target image based on the dropdown
        if "Input" in self.target_combo.currentText():
            target_image = self.image
        elif "Output1" in self.target_combo.currentText():
             target_image = self.output1
        elif "Output2" in self.target_combo.currentText():
             target_image = self.output2
        else:
            return

        # Ensure the target image is valid
        if target_image is not None:
            # Get the selected contrast enhancement method
            contrast_type = self.contrast_combo.currentText()
            enhanced_image = target_image.copy()

            if "Histogram Equalization" in contrast_type:
                # Apply histogram equalization (for grayscale images) low con stretch, high cont unchanged
                if len(target_image.shape) == 2:  # Grayscale
                    enhanced_image = cv2.equalizeHist(target_image)
                else:  # Color
                    ycrcb = cv2.cvtColor(target_image, cv2.COLOR_BGR2YCrCb)
                    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                    enhanced_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

            elif "CLAHE" in contrast_type:  # Contrast Limited Adaptive Histogram Equalization divide into small regions and enhance on each tile
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                if len(target_image.shape) == 2:  # Grayscale
                    enhanced_image = clahe.apply(target_image)
                else:  # Color
                    ycrcb = cv2.cvtColor(target_image, cv2.COLOR_BGR2YCrCb)
                    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
                    enhanced_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

            elif "Gamma Correction" in contrast_type:
                gamma = 1.5  # Example value for gamma correction Gamma > 1: Darkens the image6
                inv_gamma = 1.0 / gamma
                table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced_image = cv2.LUT(target_image, table)

            # Update the selected viewport with the enhanced image
            self.update_viewport(self.active_viewport, enhanced_image)

    
    def update_viewport(self, viewport , modified_image):
            """Update the currently active viewport with the modified image."""
            if viewport == self.input_view:
               self.image = modified_image
               self.display_image(self.input_view, self.image)
            elif viewport == self.output1_view:
                 self.output1 = modified_image
                 self.display_image(self.output1_view, self.output1)
            elif viewport == self.output2_view:
                 self.output2 = modified_image
                 self.display_image(self.output2_view, self.output2)

    def show_histogram(self, image, title):
        """Show histogram of the given image."""
        if image is not None:
            plt.figure(title)
            plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
            plt.title(title)
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.show()

class CalculationWindow(QMainWindow):
    def __init__(self, parent, mode="CNR", image=None):
        super().__init__()
        self.parent = parent
        self.mode = mode
        self.image = image
        self.setWindowTitle(f"{mode} Calculation")
        self.setGeometry(200, 200, 800, 800)

        # Check if the image is valid
        if self.image is None:
            QMessageBox.warning(self, "Error", "No active image to process!")
            self.close()
            return

        self.rois = []  # List to store ROI rectangles

        # Main Layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Graphics View and Scene
        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self.scene, parent=self)
        self.view.setScene(self.scene)
        self.view.roi_drawn.connect(self.add_roi)
        self.layout.addWidget(self.view)

        # Display the image
        self.display_image()

        # Buttons
        button_layout = QHBoxLayout()
        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.clicked.connect(self.calculate)
        button_layout.addWidget(self.calculate_btn)

        self.reset_btn = QPushButton("Reset ROIs")
        self.reset_btn.clicked.connect(self.reset_rois)
        button_layout.addWidget(self.reset_btn)

        self.layout.addLayout(button_layout)

    def display_image(self):
        """Display the image in the QGraphicsView."""
        height, width = self.image.shape[:2]
        bytes_per_line = width
        q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.scene.clear()  # Clear previous content
        self.scene.addPixmap(pixmap)
        self.view.setSceneRect(0, 0, width, height)

    def add_roi(self, rect):
        """Add ROI to the list and draw it."""
        print(f"ROI added: {rect}")
        self.rois.append(rect)
        pen_color = Qt.green if len(self.rois) == 1 else Qt.blue
        self.scene.addRect(rect, QPen(pen_color))

    def reset_rois(self):
        """Reset all drawn ROIs."""
        self.rois.clear()
        self.scene.clear()
        self.display_image()

    def calculate(self):
        """Perform the calculation based on the ROIs."""
        if self.mode == "CNR" and len(self.rois) != 3:
            QMessageBox.warning(self, "Error", "You need exactly 3 ROIs for CNR calculation.")
            return
        elif self.mode == "SNR" and len(self.rois) != 2:
            QMessageBox.warning(self, "Error", "You need exactly 2 ROIs for SNR calculation.")
            return

        # Extract pixel values for each ROI
        roi_data = []
        for rect in self.rois:
            x, y, w, h = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
            roi = self.image[y:y + h, x:x + w]
            roi_data.append((np.mean(roi), np.std(roi)))

        if self.mode == "CNR":
            # CNR requires 3 ROIs: Signal 1, Signal 2, and Noise
            signal1_mean = roi_data[0][0]
            signal2_mean = roi_data[1][0]
            noise_std = roi_data[2][1]
            if noise_std == 0:
                QMessageBox.warning(self, "Error", "Noise standard deviation is zero.")
            else:
                cnr = abs(signal1_mean - signal2_mean) / noise_std
                QMessageBox.information(self, "CNR", f"CNR: {cnr:.2f}")

        elif self.mode == "SNR":
            # SNR requires 2 ROIs: Signal and Noise
            signal_mean = roi_data[0][0]
            noise_std = roi_data[1][1]
            if noise_std == 0:
                QMessageBox.warning(self, "Error", "Noise standard deviation is zero.")
            else:
                snr = signal_mean / noise_std
                QMessageBox.information(self, "SNR", f"SNR: {snr:.2f}")

class CustomGraphicsView(QGraphicsView):
    roi_drawn = pyqtSignal(QRectF)  # Signal to emit when an ROI is drawn

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.current_start = None

    def mousePressEvent(self, event):
        """Capture the start of an ROI."""
        scene_pos = self.mapToScene(event.pos())
        self.current_start = QPointF(scene_pos.x(), scene_pos.y())

    def mouseReleaseEvent(self, event):
        """Capture the end of an ROI and emit a signal."""
        if not self.current_start:
            return

        scene_pos = self.mapToScene(event.pos())
        rect = QRectF(self.current_start, scene_pos).normalized()  # Normalize ensures valid rectangle
        if rect.width() > 0 and rect.height() > 0:  # Avoid zero-sized rectangles
            self.roi_drawn.emit(rect)
        self.current_start = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
