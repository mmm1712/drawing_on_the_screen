import time, math, sys, os
from datetime import datetime

import numpy as np
import cv2, mediapipe as mp

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QRectF, pyqtSlot
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QPainterPath, QImage
from PyQt6.QtWidgets import QApplication, QWidget

CAM_INDEX = 0
CAP_W, CAP_H = 640, 480

class OneEuro:
    def __init__(self, min_cutoff=1.3, beta=0.03, dcutoff=1.0):
        self.min_cutoff, self.beta, self.dcutoff = min_cutoff, beta, dcutoff
        self.x_prev = None; self.dx_prev = None; self.t_prev = None
    def _alpha(self, cutoff, dt):
        return 1.0/(1.0+(1.0/(2*math.pi*cutoff*dt)))
    def __call__(self, x):
        t = time.time()
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, x
            return x
        dt = max(1e-3, t-self.t_prev); self.t_prev = t
        dx = (x-self.x_prev)/dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = dx if self.dx_prev is None else a_d*dx + (1-a_d)*self.dx_prev
        self.dx_prev = dx_hat
        cutoff = self.min_cutoff + self.beta*abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a*x + (1-a)*self.x_prev
        self.x_prev = x_hat
        return x_hat


PINCH_ON, PINCH_OFF = 0.045, 0.060


GLASS_BG = QColor(18,18,20,155)
GLASS_BORDER = QColor(255,255,255,40)
KEY_TEXT = QColor(245,245,245)
KEY_TEXT_DIM = QColor(210,210,210)
HOVER_BORDER = QColor(255,255,255,90)


TOOL_H = 64
BRUSH_COLORS = [
    QColor(242, 99, 123),  # pink
    QColor(82, 125, 243),  # blue
    QColor(87, 219, 133),  # green
    QColor(255, 212, 96),  # yellow
    QColor(255, 255, 255), # white
    QColor(60, 60, 60),    # charcoal
]
DEFAULT_BRUSH_INDEX = 4
DEFAULT_BRUSH_SIZE = 6
MAX_BRUSH_SIZE = 30
MIN_BRUSH_SIZE = 1

CURSOR_SIZE = 18
CURSOR_THICK = 3
KEY_RADIUS = 14

def rounded_rect_path(r: QRect, radius: int):
    path = QPainterPath()
    rf = QRectF(r)
    path.addRoundedRect(rf, float(radius), float(radius))
    return path

class HandWorker(QThread):
    update = pyqtSignal(dict)  # {'idx':(x,y), 'pinch':bool}

    def __init__(self):
        super().__init__()

        self.fx = OneEuro(min_cutoff=2.2, beta=0.08, dcutoff=1.0)
        self.fy = OneEuro(min_cutoff=2.2, beta=0.08, dcutoff=1.0)

        self.running = True
        self.pinched = False

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.65
        )
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            payload = {'idx': None, 'pinch': False}

            if res.multi_hand_landmarks:
                lms = res.multi_hand_landmarks[0].landmark

                idx = (self.fx(lms[8].x), self.fy(lms[8].y))

                thumb, index = lms[4], lms[8]
                d_pinch = math.hypot(thumb.x-index.x, thumb.y-index.y)

                if not self.pinched and d_pinch < PINCH_ON:
                    self.pinched = True
                elif self.pinched and d_pinch > PINCH_OFF:
                    self.pinched = False

                payload.update({'idx': idx, 'pinch': self.pinched})

            self.update.emit(payload)
            self.msleep(16)

        self.cap.release()

    def stop(self):
        self.running = False

class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # hand state
        self.idx = None
        self.pinched = False
        self.last_pinch = False

        # pinch stability
        self._pinch_last_true_ms = 0.0
        self.PINCH_GRACE_MS = 80

        # draw state
        self.brush_index = DEFAULT_BRUSH_INDEX
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.strokes = []         # [{'pts':[(x,y)...], 'color':QColor, 'size':int}]
        self.current_stroke = None

        # worker thread
        self.worker = HandWorker()
        self.worker.update.connect(self.on_hand)
        self.worker.start()

        # fonts
        self.font_large = QFont("", 24)
        self.font_medium = QFont("", 22)
        self.font_small = QFont("", 18)

        self._in_tick = False


        self.logic_timer = QTimer(self)
        self.logic_timer.timeout.connect(self.tick)
        self.logic_timer.start(16)

        self.repaint_timer = QTimer(self)
        self.repaint_timer.timeout.connect(self.update)
        self.repaint_timer.start(16)

    @pyqtSlot(dict)
    def on_hand(self, data):
        self.idx = data.get('idx')
        self.pinched = bool(data.get('pinch', False))


    def toolbar_rects(self):
        W, H = self.width(), self.height()
        util_w = int(W*0.88)
        util_x = (W - util_w)//2
        util_y = int(0.08*H)
        util_rect = QRect(util_x, util_y, util_w, TOOL_H)

        pad = 10
        x = util_x + pad
        y = util_y + 6

        # size label + - +
        size_lbl = QRect(x, y, 82, TOOL_H-12); x += size_lbl.width()+6
        minus_btn = QRect(x, y, 34, TOOL_H-12); x += minus_btn.width()+4
        plus_btn  = QRect(x, y, 34, TOOL_H-12); x += plus_btn.width()+12

        # colors
        color_rects = []
        circle_d = TOOL_H-16
        for _ in BRUSH_COLORS:
            r = QRect(x, y, circle_d, circle_d)
            color_rects.append(r)
            x += circle_d + 8

        # Clear + Save
        clr_btn = QRect(x+6, y, 74, TOOL_H-12); x += clr_btn.width()+6
        save_btn = QRect(x+6, y, 74, TOOL_H-12)

        return util_rect, size_lbl, minus_btn, plus_btn, color_rects, clr_btn, save_btn


    def clear_canvas(self):
        self.strokes.clear()
        self.current_stroke = None

    def save_png(self):

        W, H = self.width(), self.height()
        img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(0,0,0,0))
        qp = QPainter(img)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for s in self.strokes:
            pts = s['pts']
            if len(pts) < 2:
                continue
            pen = QPen(s['color'])
            pen.setWidth(int(s['size']))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            qp.setPen(pen)
            for i in range(1, len(pts)):
                x1,y1 = pts[i-1]; x2,y2 = pts[i]
                qp.drawLine(int(x1), int(y1), int(x2), int(y2))

        qp.end()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.expanduser(f"~/Desktop/air_draw_{ts}.png")
        img.save(out_path)
        print(f"Saved: {out_path}")

    def start_stroke(self, x, y):
        self.current_stroke = {
            'pts': [(float(x), float(y))],
            'color': BRUSH_COLORS[self.brush_index],
            'size': self.brush_size
        }
        self.strokes.append(self.current_stroke)

    def extend_stroke(self, x, y):

        if self.current_stroke is None:
            return

        x = float(x); y = float(y)
        px, py = self.current_stroke['pts'][-1]
        dist = math.hypot(x - px, y - py)

        steps = max(1, int(dist / 3.0))
        for i in range(1, steps + 1):
            ix = px + (x - px) * i / steps
            iy = py + (y - py) * i / steps
            self.current_stroke['pts'].append((ix, iy))

    def end_stroke(self):
        self.current_stroke = None

    def tick(self):

        if self._in_tick:
            return
        self._in_tick = True

        try:
            util_rect, size_lbl, minus_btn, plus_btn, color_rects, clr_btn, save_btn = self.toolbar_rects()

            hover_util = None
            hx = hy = None

            if self.idx is not None:
                hx = int(self.idx[0] * self.width())
                hy = int(self.idx[1] * self.height())

           
                if util_rect.contains(hx, hy):
                    if size_lbl.contains(hx, hy):  hover_util = ('size', size_lbl)
                    elif minus_btn.contains(hx, hy): hover_util = ('minus', minus_btn)
                    elif plus_btn.contains(hx, hy):  hover_util = ('plus', plus_btn)
                    elif clr_btn.contains(hx, hy):   hover_util = ('clear', clr_btn)
                    elif save_btn.contains(hx, hy):  hover_util = ('save', save_btn)
                    else:
                        for i, cr in enumerate(color_rects):
                            if cr.contains(hx, hy):
                                hover_util = ('color', (i, cr))
                                break

                pinch_rise = self.pinched and not self.last_pinch

              
                if pinch_rise and hover_util is not None:
                    action = hover_util[0]
                    if action == 'minus':
                        self.brush_size = max(MIN_BRUSH_SIZE, self.brush_size - 1)
                    elif action == 'plus':
                        self.brush_size = min(MAX_BRUSH_SIZE, self.brush_size + 1)
                    elif action == 'clear':
                        QTimer.singleShot(0, self.clear_canvas)
                    elif action == 'save':
                        QTimer.singleShot(0, self.save_png)
                    elif action == 'color':
                        self.brush_index = hover_util[1][0]

                else:
            
                    now_ms = time.time() * 1000.0

                    if self.pinched:
                        self._pinch_last_true_ms = now_ms
                        if self.current_stroke is None:
                  
                            if pinch_rise:
                                self.start_stroke(hx, hy)
                            else:
                              
                                self.start_stroke(hx, hy)
                        else:
                            self.extend_stroke(hx, hy)
                    else:
                      
                        if self.current_stroke is not None:
                            if (now_ms - self._pinch_last_true_ms) > self.PINCH_GRACE_MS:
                                self.end_stroke()

 
            self.last_pinch = self.pinched

        finally:
            self._in_tick = False


    def draw_glass_panel(self, p: QPainter, r: QRect, radius=16):
        path = rounded_rect_path(r, radius)
        rf = QRectF(r)
        grad = QLinearGradient(rf.topLeft(), rf.bottomLeft())
        grad.setColorAt(0.0, GLASS_BG)
        grad.setColorAt(1.0, QColor(GLASS_BG.red(), GLASS_BG.green(), GLASS_BG.blue(), min(255, GLASS_BG.alpha()+10)))
        p.fillPath(path, grad)

    def draw_key(self, p: QPainter, r: QRect, label: str, is_hover: bool):
        rf = QRectF(r)
        body = QLinearGradient(rf.topLeft(), rf.bottomLeft())
        baseA = 175 if is_hover else 150
        body.setColorAt(0.0, QColor(28,28,32, baseA))
        body.setColorAt(1.0, QColor(24,24,28, baseA))
        path = rounded_rect_path(r, KEY_RADIUS)
        p.fillPath(path, body)
        p.setPen(QPen(HOVER_BORDER if is_hover else GLASS_BORDER, 2 if is_hover else 1))
        p.drawPath(path)
        p.setPen(KEY_TEXT if is_hover else KEY_TEXT_DIM)
        p.setFont(self.font_medium)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, label)


    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        W, H = self.width(), self.height()
        hx, hy = (int(self.idx[0]*W), int(self.idx[1]*H)) if self.idx else (None, None)

        if self.strokes:
            for s in self.strokes:
                pts = s['pts']
                if len(pts) < 2:
                    continue
                pen = QPen(s['color'])
                pen.setWidth(int(s['size']))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(pen)
                for i in range(1, len(pts)):
                    x1,y1 = pts[i-1]; x2,y2 = pts[i]
                    p.drawLine(int(x1), int(y1), int(x2), int(y2))


        util_rect, size_lbl, minus_btn, plus_btn, color_rects, clr_btn, save_btn = self.toolbar_rects()
        self.draw_glass_panel(p, util_rect, 12)

        # size controls
        self.draw_key(p, size_lbl, f"size {self.brush_size}", False)
        self.draw_key(p, minus_btn, "–", False)
        self.draw_key(p, plus_btn,  "+", False)

        # colors
        for i, r in enumerate(color_rects):
            path = rounded_rect_path(r, r.height()//2)
            p.fillPath(path, BRUSH_COLORS[i])
            active = (i == self.brush_index)
            p.setPen(QPen(QColor(255,255,255,200) if active else GLASS_BORDER, 2 if active else 1))
            p.drawPath(path)

        # clear/save
        self.draw_key(p, clr_btn, "Clear", False)
        self.draw_key(p, save_btn, "Save", False)

        # minimal cursor ring (no glow)
        if self.idx is not None:
            p.setPen(QPen(QColor(255,255,255,230), CURSOR_THICK))
            p.drawEllipse(hx-CURSOR_SIZE, hy-CURSOR_SIZE, 2*CURSOR_SIZE, 2*CURSOR_SIZE)
            p.setPen(QPen(QColor(255,255,255,160), 1))
            p.drawLine(hx-CURSOR_SIZE-5, hy, hx+CURSOR_SIZE+5, hy)
            p.drawLine(hx, hy-CURSOR_SIZE-5, hx, hy+CURSOR_SIZE-5)

    # clean up
    def closeEvent(self, ev):
        self.worker.stop()
        self.worker.wait(500)
        return super().closeEvent(ev)

# main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = Overlay()
    overlay.setWindowState(Qt.WindowState.WindowFullScreen)
    overlay.show()
    sys.exit(app.exec())
