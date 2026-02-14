"""
CAE Stress Prediction Model Training Client
CAE应力预测模型训练C端客户端

Generated from OpenSpec: cae_ml_client_ui v1.0.0
PyQt6 Desktop Application for CAE Model Training

Features:
- Interactive UI for model training configuration
- Real-time training log display
- Matplotlib embedded visualization for loss curves
- Multi-threaded training to prevent UI blocking
- File path validation and error handling

Usage:
    python cae_ml_client.py

Dependencies:
    pip install -r requirements.txt
"""

import sys
import os
import time
import json
from typing import Optional, Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QSplitter, QGroupBox, QLabel, QLineEdit,
    QPushButton, QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget,
    QFileDialog, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import the model from existing module
from model import CAEStressPredictionMLP, create_model


# =============================================================================
# Training Worker Thread (Signal-based communication with UI)
# =============================================================================

class TrainingSignals(QObject):
    """Signals for thread-safe communication between training thread and UI"""
    log_update = pyqtSignal(str)           # Training log messages
    loss_update = pyqtSignal(dict)         # Loss metrics update
    train_complete = pyqtSignal(bool, str) # Training completion status
    progress_update = pyqtSignal(int)      # Progress percentage


class TrainingWorker(QThread):
    """
    Training worker thread to run model training without blocking UI.
    
    This thread handles the complete training pipeline including:
    - Data loading and preprocessing
    - Model training loop
    - Validation and metrics calculation
    - Model saving
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.signals = TrainingSignals()
        self._is_running = True
        self.device = None
        self.model = None
        
    def stop(self):
        """Signal the training to stop gracefully"""
        self._is_running = False
        self.signals.log_update.emit("\n>>> 收到停止信号，将在当前epoch结束后停止...")
        
    def load_data(self, data_path: str) -> tuple:
        """
        Load and preprocess training data from CSV file.
        
        Expected CSV format:
        - Features: length, width, thickness, E, nu, load
        - Target: stress
        
        Args:
            data_path: Path to the CSV file
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val) as torch tensors
        """
        try:
            df = pd.read_csv(data_path)
            self.signals.log_update.emit(f"数据文件加载成功: {data_path}")
            self.signals.log_update.emit(f"数据形状: {df.shape}")
            self.signals.log_update.emit(f"列名: {list(df.columns)}")
            
            # Determine feature columns (expecting 6 features)
            feature_cols = ['length', 'width', 'thickness', 'E', 'nu', 'load']
            target_col = 'stress'
            
            # Check if columns exist, if not try to infer
            available_cols = [c.lower().replace('_', '') for c in df.columns]
            
            # Map column names
            X_cols = []
            for col in df.columns:
                col_lower = col.lower().replace('_', '').replace('mm', '').replace('gpa', '')
                if any(f in col_lower for f in ['length', 'width', 'thickness', 'young', 'elastic', 'poisson', 'nu', 'load', 'force']):
                    if col_lower not in ['stress', 'target']:
                        X_cols.append(col)
            
            # If we found exactly 6 feature columns, use them
            if len(X_cols) == 6:
                feature_cols = X_cols
            else:
                # Use all columns except the last one as features
                feature_cols = list(df.columns[:-1])
                
            # Target column
            y_col = None
            for col in df.columns:
                if 'stress' in col.lower() or 'target' in col.lower() or col == df.columns[-1]:
                    y_col = col
                    break
            
            if y_col is None:
                y_col = df.columns[-1]
                
            self.signals.log_update.emit(f"特征列: {feature_cols}")
            self.signals.log_update.emit(f"目标列: {y_col}")
            
            X = df[feature_cols].values.astype(np.float32)
            y = df[y_col].values.astype(np.float32).reshape(-1, 1)
            
            # Split data (80% train, 20% val)
            n_samples = len(X)
            n_train = int(0.8 * n_samples)
            
            indices = np.random.permutation(n_samples)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]
            
            X_train = torch.FloatTensor(X[train_idx])
            y_train = torch.FloatTensor(y[train_idx])
            X_val = torch.FloatTensor(X[val_idx])
            y_val = torch.FloatTensor(y[val_idx])
            
            self.signals.log_update.emit(f"训练集: {len(X_train)} 样本")
            self.signals.log_update.emit(f"验证集: {len(X_val)} 样本")
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            raise RuntimeError(f"数据加载失败: {str(e)}")
    
    def calculate_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Calculate regression metrics"""
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        mse = torch.mean((y_true - y_pred) ** 2).item()
        rmse = np.sqrt(mse)
        
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if not self._is_running:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    def validate(self, model: nn.Module, dataloader: DataLoader, 
                 criterion: nn.Module) -> tuple:
        """Validate model"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                all_preds.append(output.cpu())
                all_targets.append(target.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.calculate_metrics(all_targets, all_preds)
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        return avg_loss, metrics, all_targets.numpy(), all_preds.numpy()
    
    def run(self):
        """Main training loop executed in separate thread"""
        try:
            # Load data
            self.signals.log_update.emit("=" * 60)
            self.signals.log_update.emit("开始加载数据...")
            X_train, y_train, X_val, y_val = self.load_data(self.config['data_path'])
            
            # Create model
            self.signals.log_update.emit("\n创建模型...")
            self.model, self.device = create_model()
            self.signals.log_update.emit(f"使用设备: {self.device}")
            
            # Setup optimizer and loss
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['lr'],
                weight_decay=1e-5
            )
            criterion = nn.SmoothL1Loss()
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config['batch_size']
            )
            
            # Training loop
            self.signals.log_update.emit(f"\n开始训练 (最大 {self.config['max_epochs']} epochs)...")
            self.signals.log_update.emit("-" * 60)
            
            history = {
                'train_loss': [],
                'val_loss': [],
                'mae': [],
                'r2': []
            }
            best_val_loss = float('inf')
            
            for epoch in range(self.config['max_epochs']):
                if not self._is_running:
                    self.signals.log_update.emit("\n训练已手动停止")
                    break
                
                start_time = time.time()
                
                # Train
                train_loss = self.train_epoch(self.model, train_loader, optimizer, criterion)
                
                # Validate
                val_loss, metrics, y_true, y_pred = self.validate(self.model, val_loader, criterion)
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['mae'].append(metrics['mae'])
                history['r2'].append(metrics['r2'])
                
                epoch_time = time.time() - start_time
                
                # Log progress
                log_msg = (f"Epoch [{epoch+1}/{self.config['max_epochs']}] "
                          f"Time: {epoch_time:.2f}s | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | "
                          f"MAE: {metrics['mae']:.4f} | "
                          f"R²: {metrics['r2']:.4f}")
                self.signals.log_update.emit(log_msg)
                
                # Update progress
                progress = int((epoch + 1) / self.config['max_epochs'] * 100)
                self.signals.progress_update.emit(progress)
                
                # Send loss update for plotting
                self.signals.loss_update.emit({
                    'epoch': epoch + 1,
                    'train_loss': history['train_loss'],
                    'val_loss': history['val_loss'],
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'metrics': metrics
                })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = self.config['model_path']
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'metrics': metrics
                    }, model_path)
                    self.signals.log_update.emit(f"  -> 保存最佳模型 (val_loss: {val_loss:.6f})")
            
            self.signals.log_update.emit("-" * 60)
            self.signals.log_update.emit("训练完成!")
            self.signals.log_update.emit(f"最佳验证损失: {best_val_loss:.6f}")
            self.signals.log_update.emit(f"模型保存路径: {self.config['model_path']}")
            self.signals.train_complete.emit(True, "训练成功完成")
            
        except Exception as e:
            error_msg = f"训练过程中发生错误: {str(e)}"
            self.signals.log_update.emit(f"\n[错误] {error_msg}")
            self.signals.train_complete.emit(False, error_msg)


# =============================================================================
# Matplotlib Canvas for Embedding in PyQt
# =============================================================================

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in PyQt widgets"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def clear(self):
        """Clear the canvas"""
        self.axes.clear()
        
    def draw_plot(self):
        """Redraw the canvas"""
        self.fig.tight_layout()
        self.draw()


# =============================================================================
# Main Application Window
# =============================================================================

class CAEMLClient(QMainWindow):
    """
    Main window for CAE Stress Prediction Model Training Client.
    
    This class implements the complete UI layout and interaction logic
    as specified in the OpenSpec v1 PyQtUI configuration.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAE应力预测模型训练工具")
        self.setMinimumSize(800, 600)
        self.resize(1200, 800)
        
        # Center window on screen
        self._center_window()
        
        # Initialize training thread
        self.train_thread: Optional[TrainingWorker] = None
        
        # Setup UI
        self._setup_ui()
        self._apply_styles()
        
    def _center_window(self):
        """Center the window on the screen"""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
        
    def _setup_ui(self):
        """Setup the main UI layout"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel: Configuration
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        splitter.setSizes([350, 850])
        
        # Right panel: Results
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
    def _create_left_panel(self) -> QWidget:
        """Create the left configuration panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Group 1: Data Configuration
        data_group = QGroupBox("数据配置")
        data_layout = QFormLayout(data_group)
        
        # Training data path
        data_layout.addRow(QLabel("训练数据路径："))
        
        self.le_train_path = QLineEdit()
        self.le_train_path.setPlaceholderText("请选择CAE训练数据CSV文件")
        data_layout.addRow(self.le_train_path)
        
        self.btn_select_train = QPushButton("浏览")
        self.btn_select_train.clicked.connect(self.select_train_data)
        data_layout.addRow(self.btn_select_train)
        
        # Model save path
        data_layout.addRow(QLabel("模型保存路径："))
        
        self.le_model_path = QLineEdit("./models/cae_model.pth")
        data_layout.addRow(self.le_model_path)
        
        self.btn_select_model = QPushButton("浏览")
        self.btn_select_model.clicked.connect(self.select_model_path)
        data_layout.addRow(self.btn_select_model)
        
        layout.addWidget(data_group)
        
        # Group 2: Training Parameters
        param_group = QGroupBox("训练参数")
        param_layout = QFormLayout(param_group)
        
        # Batch size
        param_layout.addRow(QLabel("批次大小："))
        self.sb_batch_size = QSpinBox()
        self.sb_batch_size.setRange(8, 128)
        self.sb_batch_size.setValue(64)
        self.sb_batch_size.setSingleStep(8)
        param_layout.addRow(self.sb_batch_size)
        
        # Max epochs
        param_layout.addRow(QLabel("最大轮数："))
        self.sb_max_epochs = QSpinBox()
        self.sb_max_epochs.setRange(10, 500)
        self.sb_max_epochs.setValue(200)
        self.sb_max_epochs.setSingleStep(10)
        param_layout.addRow(self.sb_max_epochs)
        
        # Learning rate
        param_layout.addRow(QLabel("学习率："))
        self.dsb_lr = QDoubleSpinBox()
        self.dsb_lr.setRange(0.0001, 0.01)
        self.dsb_lr.setValue(0.0005)
        self.dsb_lr.setDecimals(4)
        self.dsb_lr.setSingleStep(0.0001)
        param_layout.addRow(self.dsb_lr)
        
        layout.addWidget(param_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Group 3: Control Buttons
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_start_train = QPushButton("开始训练")
        self.btn_start_train.setObjectName("btn_start_train")
        self.btn_start_train.clicked.connect(self.start_model_training)
        btn_layout.addWidget(self.btn_start_train)
        
        self.btn_stop_train = QPushButton("停止训练")
        self.btn_stop_train.setObjectName("btn_stop_train")
        self.btn_stop_train.clicked.connect(self.stop_model_training)
        self.btn_stop_train.setEnabled(False)
        btn_layout.addWidget(self.btn_stop_train)
        
        layout.addWidget(btn_widget)
        layout.addStretch()
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right results panel with tabs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Tab 1: Training Log
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.te_log = QTextEdit()
        self.te_log.setReadOnly(True)
        self.te_log.setPlaceholderText("训练日志将显示在这里...")
        font = QFont("Consolas", 10)
        self.te_log.setFont(font)
        log_layout.addWidget(self.te_log)
        
        tab_widget.addTab(log_tab, "训练日志")
        
        # Tab 2: Visualization
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        # Loss curve
        viz_layout.addWidget(QLabel("训练损失曲线"))
        self.loss_canvas = MplCanvas(self, width=8, height=4)
        viz_layout.addWidget(self.loss_canvas)
        
        # Stress prediction comparison
        viz_layout.addWidget(QLabel("应力预测结果对比"))
        self.stress_canvas = MplCanvas(self, width=8, height=4)
        viz_layout.addWidget(self.stress_canvas)
        
        tab_widget.addTab(viz_tab, "结果可视化")
        
        return panel
    
    def _apply_styles(self):
        """Apply CSS styles to the UI components"""
        self.setStyleSheet("""
            QWidget {
                font-family: "Microsoft YaHei", sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                font-weight: bold;
                margin-top: 10px;
                padding-top: 10px;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
                background-color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton#btn_start_train {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#btn_start_train:hover {
                background-color: #45a049;
            }
            QPushButton#btn_stop_train {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#btn_stop_train:hover {
                background-color: #da190b;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def select_train_data(self):
        """Open file dialog to select training data CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择训练数据文件",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if file_path:
            self.le_train_path.setText(file_path)
            self._log_message(f"已选择训练数据: {file_path}")
    
    def select_model_path(self):
        """Open folder dialog to select model save path"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择模型保存路径",
            "./models/cae_model.pth",
            "PyTorch Model (*.pth);;All Files (*.*)"
        )
        if file_path:
            self.le_model_path.setText(file_path)
            self._log_message(f"已设置模型保存路径: {file_path}")
    
    def start_model_training(self):
        """Start the model training process"""
        # Validate inputs
        data_path = self.le_train_path.text().strip()
        model_path = self.le_model_path.text().strip()
        
        if not data_path:
            QMessageBox.warning(self, "警告", "请选择训练数据文件!")
            return
        
        if not os.path.exists(data_path):
            QMessageBox.critical(self, "错误", f"数据文件不存在: {data_path}")
            return
        
        if not data_path.endswith('.csv'):
            QMessageBox.warning(self, "警告", "请选择CSV格式的数据文件!")
            return
        
        # Ensure model directory exists
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法创建模型目录: {str(e)}")
                return
        
        # Get training configuration
        config = {
            'data_path': data_path,
            'model_path': model_path,
            'batch_size': self.sb_batch_size.value(),
            'max_epochs': self.sb_max_epochs.value(),
            'lr': self.dsb_lr.value()
        }
        
        # Update UI state
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.progress_bar.setValue(0)
        self.te_log.clear()
        
        # Clear plots
        self.loss_canvas.clear()
        self.loss_canvas.draw_plot()
        self.stress_canvas.clear()
        self.stress_canvas.draw_plot()
        
        # Start training thread
        self.train_thread = TrainingWorker(config)
        self.train_thread.signals.log_update.connect(self._log_message)
        self.train_thread.signals.loss_update.connect(self._update_plots)
        self.train_thread.signals.train_complete.connect(self._training_finished)
        self.train_thread.signals.progress_update.connect(self._update_progress)
        self.train_thread.start()
        
        self._log_message("=" * 60)
        self._log_message("CAE应力预测模型训练客户端")
        self._log_message("=" * 60)
    
    def stop_model_training(self):
        """Stop the model training process"""
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.stop()
            self._log_message("\n>>> 正在停止训练...")
    
    def _log_message(self, message: str):
        """Append message to log text edit"""
        self.te_log.append(message)
        # Auto-scroll to bottom
        scrollbar = self.te_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _update_progress(self, value: int):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def _update_plots(self, data: dict):
        """Update visualization plots with new data"""
        # Update loss curve
        self.loss_canvas.clear()
        epochs = range(1, len(data['train_loss']) + 1)
        self.loss_canvas.axes.plot(epochs, data['train_loss'], 'b-', label='Train Loss')
        self.loss_canvas.axes.plot(epochs, data['val_loss'], 'r-', label='Val Loss')
        self.loss_canvas.axes.set_xlabel('Epoch')
        self.loss_canvas.axes.set_ylabel('Loss')
        self.loss_canvas.axes.set_title('Training and Validation Loss')
        self.loss_canvas.axes.legend()
        self.loss_canvas.axes.grid(True, alpha=0.3)
        self.loss_canvas.draw_plot()
        
        # Update stress prediction comparison
        self.stress_canvas.clear()
        y_true = data['y_true'].flatten()
        y_pred = data['y_pred'].flatten()
        
        # Scatter plot
        self.stress_canvas.axes.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        self.stress_canvas.axes.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        self.stress_canvas.axes.set_xlabel('True Stress (MPa)')
        self.stress_canvas.axes.set_ylabel('Predicted Stress (MPa)')
        self.stress_canvas.axes.set_title(f"Stress Prediction (MAE: {data['metrics']['mae']:.4f}, R²: {data['metrics']['r2']:.4f})")
        self.stress_canvas.axes.legend()
        self.stress_canvas.axes.grid(True, alpha=0.3)
        self.stress_canvas.draw_plot()
    
    def _training_finished(self, success: bool, message: str):
        """Handle training completion"""
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.critical(self, "错误", message)
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.train_thread and self.train_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "训练正在进行中，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.train_thread.stop()
                self.train_thread.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Application entry point"""
    # Enable high DPI scaling
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = CAEMLClient()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
