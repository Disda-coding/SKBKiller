# 新增必要的导入
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal,QTimer  # 添加pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QCursor,QKeySequence,QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel,QShortcut,QVBoxLayout
import numpy as np
import os
import cv2
import pytesseract
import pandas as pd
from mss import mss
from fuzzywuzzy import process,fuzz
import sys
from PyQt5.QtGui import QRegion
from operator import itemgetter

def enhance_image(img):
    """增强图像质量的综合处理"""
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 直方图均衡化
    gray = cv2.equalizeHist(gray)
    
    # 自适应阈值
    thresh = cv2.adaptiveThreshold(gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2)
    
    # 降噪处理
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 锐化处理
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    thresh = cv2.filter2D(thresh, -1, kernel)
    
    return thresh

class CaptureWindow(QWidget):
    capture_completed = pyqtSignal(str)
    continuous_mode = False  # 新增持续模式标志
    last_result_hash = None  # 新增：防抖校验
    debounce_timer = None  # 新增：防抖定时器

    def __init__(self,questions):
        super().__init__()
        self.questions = questions  # 存储题库数据
        self.last_recognized_text = ""  # 新增：保存最近一次识别文本
        self.result_window = ResultWindow()
        self.result_window.setMaximumWidth(400)  # 设置更紧凑的宽度
        self.begin = QPoint()
        self.end = QPoint()
        self.timer = QTimer(self)  # 初始化定时器
        self.timer.timeout.connect(self.continuous_capture)
        self.init_ui()
        self.init_capture_params()
        self.selection_rect = QRect()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 新增鼠标事件穿透
        self.setFocusPolicy(Qt.NoFocus)  # 禁止获取焦点
        self.init_shortcuts()  # 新增初始化快捷方式
        self.result_window = ResultWindow()  # 新增结果窗口
        self.debounce_timer = QTimer(self)  # 新增防抖定时器
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.process_result)
        # 在CaptureWindow的__init__中初始化结果窗口时设置默认位置
        self.result_window = ResultWindow()
        screen_geo = QApplication.primaryScreen().availableGeometry()
        self.result_window.move(
            screen_geo.right() - self.result_window.width() - 20,  # 初始位置保持在右上角
            screen_geo.top() + 20
)


    def init_shortcuts(self):
        """初始化全局快捷键"""
        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.esc_shortcut.activated.connect(self.force_quit)

    def force_quit(self):
        """强制退出程序"""
        self.result_window.close()  # 新增：关闭结果窗口
        self.timer.stop()
        self.close()
        QApplication.quit()

    def init_ui(self):
        self.setWindowTitle("屏幕截图")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        
        # 获取主屏幕尺寸
        screen = QApplication.primaryScreen()
        self.screen_geo = screen.geometry()
        self.setGeometry(self.screen_geo)
        self.showFullScreen()

    def event(self, event):
        """全局事件过滤器"""
        # 允许ESC事件穿透窗口层级
        if event.type() == event.KeyPress and event.key() == Qt.Key_Escape:
            self.keyPressEvent(event)
            return True
        return super().event(event)
    
    def init_capture_params(self):
        self.corner_size = 8
        self.is_drawing = False
        self.current_corner = None
        self.cursor_map = {
            "top_left": Qt.SizeFDiagCursor,
            "top_right": Qt.SizeBDiagCursor,
            "bottom_left": Qt.SizeBDiagCursor,
            "bottom_right": Qt.SizeFDiagCursor
        }


    def get_selection_rect(self):
        return QRect(self.begin, self.end).normalized()

    def get_corners(self, rect):
        return {
            "top_left": rect.topLeft(),
            "top_right": rect.topRight(),
            "bottom_left": rect.bottomLeft(),
            "bottom_right": rect.bottomRight()
        }
    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
         # 设置文字样式
        
        # 仅当在绘制选区时显示全屏遮罩
        if not self.continuous_mode:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 160))
        
        selection = self.get_selection_rect()
        if not selection.isNull():
            # 持续模式时仅绘制选区边框
            if self.continuous_mode:
                painter.setPen(QPen(Qt.cyan, 2, Qt.DashLine))
                painter.drawRect(selection)
            else:
                # 初始模式绘制完整UI
                outer_region = QRegion(self.rect()) - QRegion(selection)
                painter.setClipRegion(outer_region)
                painter.fillRect(self.rect(), QColor(0, 0, 0, 60))
                painter.setClipping(False)
            
            # 绘制尺寸标签
            painter.setPen(Qt.white)
            label = f"{selection.width()} × {selection.height()}"  # 补充缺失的括号
            label_rect = QRect(selection.bottomRight() + QPoint(10, 5), QSize(100, 20))
            painter.drawText(label_rect, Qt.AlignLeft, label)
            # 在绘制尺寸标签后添加持续模式提示
            if self.continuous_mode:
                painter.setPen(Qt.white)
                tip_rect = QRect(selection.bottomLeft() + QPoint(10, 25), QSize(200, 20))
                painter.drawText(tip_rect, Qt.AlignLeft, "持续识别中 (ESC退出)")

            # 绘制控制点
            painter.setBrush(Qt.cyan)
            for corner in self.get_corners(selection).values():
                painter.drawRect(
                    corner.x() - self.corner_size//2,
                    corner.y() - self.corner_size//2,
                    self.corner_size,
                    self.corner_size
                )


    def mousePressEvent(self, event):
        if not self.continuous_mode: 
            if event.button() == Qt.LeftButton:
                selection = self.get_selection_rect()
                if not selection.isNull():
                    corners = self.get_corners(selection)
                    for name, pos in corners.items():
                        if self.point_in_corner(event.pos(), pos):
                            self.current_corner = name
                            self.origin_begin = self.begin
                            self.origin_end = self.end
                            return
                
                self.is_drawing = True
                self.begin = event.pos()
                self.end = event.pos()
                self.update()
            else:
                event.ignore()  # 允许事件穿透    

    def mouseMoveEvent(self, event):
        if not self.continuous_mode:  # 只在绘制选区时处理
            if self.is_drawing:
                self.end = event.pos()
                self.update()
            elif self.current_corner:
                delta = event.pos() - event.pos()
                if self.current_corner == "top_left":
                    self.begin += delta
                elif self.current_corner == "top_right":
                    self.end.setX(event.pos().x())
                    self.begin.setY(event.pos().y())
                elif self.current_corner == "bottom_left":
                    self.begin.setX(event.pos().x())
                    self.end.setY(event.pos().y())
                elif self.current_corner == "bottom_right":
                    self.end += delta
                self.update()
            else:
                self.update_cursor(event.pos())
        else:
            event.ignore()  # 允许事件穿透    
        
    def set_transparent_mode(self, enable):
        """动态设置窗口穿透属性"""
        self.setAttribute(Qt.WA_TransparentForMouseEvents, enable)
        # 保留窗口的顶层属性
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        if enable:
            flags |= Qt.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.show()  # 必须重新显示使设置生效

    def mouseReleaseEvent(self, event):
        if not self.continuous_mode:
            if event.button() == Qt.LeftButton:
                self.is_drawing = False
                self.current_corner = None
                rect = self.get_selection_rect()
                
                if not rect.isNull():
                    self.continuous_mode = True
                    self.selection_rect = rect
                    self.set_transparent_mode(True)  # 启用穿透模式
                    self.timer.start(1000)
        else:
            event.ignore()

    def continuous_capture(self):
        """持续捕获函数"""
        if self.continuous_mode:
            # 转换为绝对坐标
            screen_pos = self.mapToGlobal(self.selection_rect.topLeft())
            
            try:
                text = capture_screen_area(
                    screen_pos.x(),
                    screen_pos.y(),
                    self.selection_rect.width(),
                    self.selection_rect.height()
                )
                self.last_recognized_text = text  # 保存识别文本
                current_hash = hash(text.strip() if text else "")
                if current_hash != self.last_result_hash:
                    self.last_result_hash = current_hash
                    self.debounce_timer.start(300)  # 300ms防抖延迟

            except Exception as e:
                print(f"持续捕获错误: {str(e)}")
    def handle_result(self, text, questions):
        """带防抖的结果处理"""
        current_hash = hash(text.strip() if text else "")
        if current_hash != self.last_result_hash:
            self.last_result_hash = current_hash
            self.debounce_timer.start(300)  # 300ms防抖间隔
    def process_result(self):
        """实际处理识别结果"""
        text = self.last_recognized_text
        if text:
            # 使用实例变量中的题库数据
            results = fuzzy_search(text, self.questions)  
            display_text = self.format_results(results)
            self.result_window.set_content(display_text)
            self.result_window.show()
        else:
            self.result_window.hide()

    def format_results(self, results):
        """格式化显示内容（修复选项显示问题）"""
        if not results:
            return "没有找到匹配的题目"
        
        formatted = []
        for i, q in enumerate(results[:3], 1):
            formatted.append(f"【匹配结果{i}】")
            formatted.append(f"题目：{q['题目']}")
            
            # 修复选项处理逻辑
            options = []
            # 获取所有以'选项'开头且非空的字段（与handle_result逻辑一致）
            option_values = [v for k, v in q.items() 
                            if k.startswith('选项') and pd.notna(v)]
            
            # 生成字母选项
            for idx, value in enumerate(option_values):
                options.append(f"{chr(65+idx)}. {value}"+'\n')
            
            formatted.append("".join(options))
                # 修改答案颜色为黄色
        #     formatted.append(f"<font color='#FFD700'>答案：{q['答案']}</font>\n")  # 使用金色提高可读性
        
        # # 启用HTML格式并保留换行
        # return "<html><body style='white-space: pre'>" + "\n".join(formatted) + "</body></html>"
            formatted.append(f"答案：{q['答案']}\n")
        
        return "\n".join(formatted)



    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.force_quit()
            event.accept()
        elif event.key() == Qt.Key_C:
            self.continuous_capture()
            event.accept()
        else:
            event.ignore()

    def point_in_corner(self, point, corner):
        return QRect(
            corner.x() - self.corner_size,
            corner.y() - self.corner_size,
            self.corner_size * 2,
            self.corner_size * 2
        ).contains(point)

    def update_cursor(self, pos):
        selection = self.get_selection_rect()
        if not selection.isNull():
            corners = self.get_corners(selection)
            for name, corner in corners.items():
                if self.point_in_corner(pos, corner):
                    self.setCursor(self.cursor_map[name])
                    return
        self.setCursor(Qt.CrossCursor)


 
def load_questions(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    # 假设列名为：题目、选项A、选项B、选项C、选项D、答案
    return df.to_dict('records')


def capture_screen_area(x, y, w, h):
    """优化后的OCR函数"""
    try:
        # 设置Tesseract路径
        tesseract_path = r'C:\Program Files\Tesseract-OCR'
        pytesseract.pytesseract.tesseract_cmd = f'{tesseract_path}\\tesseract.exe'
        
              
        with mss() as sct:
            # 设置监控区域
            monitor = {"top": y, "left": x, "width": w, "height": h}
            
            # 获取截图
            img = sct.grab(monitor)
            img_array = np.array(img)

            # 图像预处理流水线
            img = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # 降噪处理
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            
            # # 超分辨率处理（可选）
            # gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # 自适应阈值
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # 设置OCR参数
            custom_config = r'''
                --oem 3
                --psm 6
                -c preserve_interword_spaces=1
                -l chi_sim
     
            '''
            
            # 执行OCR识别
            text = pytesseract.image_to_string(
                thresh,
                config=custom_config
            )
            
            return text.strip()
    
    except Exception as e:
        print(f"截图识别失败: {str(e)}")
        return ""


from fuzzywuzzy import fuzz, process
from collections import defaultdict

def fuzzy_search(query, questions, limit=5, threshold=60):
    if not query.strip():
        return []

    # 预处理查询词
    clean_query = query.strip().lower()
    exact_matches = []
    partial_results = []

    # 构建带原始索引的题目列表
    indexed_questions = [(idx, q) for idx, q in enumerate(questions)]

    # 第一轮：精确匹配 (保留大小写)
    exact_matches = [
        (q, 200, idx)  # 200分表示完全匹配
        for idx, q in indexed_questions
        if q['题目'] == query
    ]

    # 第二轮：部分匹配（如果未找到精确匹配）
    if not exact_matches:
        # 使用生成器提高内存效率
        candidates = (
            (q, fuzz.partial_ratio(clean_query, q['题目'].lower()), idx)
            for idx, q in indexed_questions
            if q['题目'] != query  # 排除精确匹配过的
        )

        # 筛选并排序
        partial_results = sorted(
            (item for item in candidates if item[1] >= threshold),
            key=lambda x: (-x[1], x[2]),  # 按分数降序，原始索引升序
        )


    # 合并结果并去重
    seen = set()
    final_results = []
    
    # 优先添加精确匹配
    for q, score, _ in exact_matches:
        text_id = id(q['题目'])
        if text_id not in seen:
            seen.add(text_id)
            final_results.append(q)
    
    # 添加部分匹配结果
    for q, score, _ in partial_results:
        text_id = id(q['题目'])
        if text_id not in seen and len(final_results) < limit:
            seen.add(text_id)
            final_results.append(q)
    
    return final_results[:limit]

class ResultWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)  # 无边框窗口
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setFocusPolicy(Qt.StrongFocus)  # 新增焦点策略
        # self.content_label.setTextFormat(Qt.RichText)  # 启用富文本解析
        # self.content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 允许选择文本
        self.dragging = False
        self.offset = QPoint(0, 0)
        
      # 修正后的样式表
        self.setStyleSheet("""
            QWidget {
                background: rgba(255, 255, 255, 0.9);
                border: 2px solid #1E90FF;
                border-radius: 10px;
                padding: 15px;
                min-width: 500px;
                max-width: 1000px;
            }
            QLabel {
                font-size: 12pt;
                color: #333333;
                qproperty-wordWrap: 1;
            }
        """)
          # 布局设置
        self.layout = QVBoxLayout()
        self.content_label = QLabel()
        self.content_label.setAlignment(Qt.AlignTop)  # 顶部对齐
        self.layout.addWidget(self.content_label)
        self.setLayout(self.layout)
    def keyPressEvent(self, event):
        """ESC键退出程序"""
        if event.key() == Qt.Key_Escape:
            QApplication.instance().quit()
        super().keyPressEvent(event)
    # 鼠标事件处理实现拖动
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            # 修改为PyQt5兼容的写法
            self.offset = event.globalPos() - self.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() & Qt.LeftButton:
            # 修改为PyQt5兼容的写法
            self.move(event.globalPos() - self.offset)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)

    def set_content(self, text):
        self.content_label.setText(text)
        self.adjustSize()  # 自动调整窗口大小





# 修改main函数
# Excel题库 题目 答案 选项 A	选项 B	选项 C	选项 D	选项E	选项F	

def main():
    try:
        questions = load_questions("人工智能竞赛.xlsx")
        app = QApplication(sys.argv)
        window = CaptureWindow(questions)  # 传入题库数据
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"初始化失败: {str(e)}")



if __name__ == "__main__":
    main()
