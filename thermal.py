#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class MileseeyThermalNode:
    def __init__(self):
        rospy.init_node('thermal_processor', anonymous=True)
        self.bridge = CvBridge()

        # Параметры из задания
        # 1. Топик для термо-изображения и матрицы температур
        self.image_pub = rospy.Publisher('/thermal/image_colored', Image, queue_size=10)
        self.matrix_pub = rospy.Publisher('/thermal/temperature_matrix', Image, queue_size=10)
        # 2. Топик для debug-изображения (с min/max/center)
        self.debug_pub = rospy.Publisher('/thermal/debug_view', Image, queue_size=10)

        # Подключение к камере
        self.cap = cv2.VideoCapture(0)

        # Настройка формата Y16 (16-bit raw)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Отключаем автоконвертацию в RGB

        # Разрешение TR256i
        self.width = 256
        self.height = 192
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)


        self.timer = rospy.Timer(rospy.Duration(0.1), self.process_frame)

    def raw_to_celsius(self, raw_frame):

        temp_c = raw_frame.astype(np.float32) /64 - 273.15

        # Если значения кажутся странными (например -270), значит камера шлет
        # данные в другом диапазоне. Попробуйте просто: temp_c = raw_frame.astype(np.float32) / 10.0
        return temp_c

    def process_frame(self, event):
        ret, frame = self.cap.read()
        if not ret: return

        try:
            # Исправляем ошибку reshape и лечим "белый экран" (byteswap)
            # 98304 байта -> 49152 пикселя (uint16) -> 192x256
            raw_16bit = frame.flatten().view(np.uint16).byteswap().reshape((self.height, self.width))

            # 1. Получаем матрицу температур (в градусах Цельсия)
            temp_matrix = self.raw_to_celsius(raw_16bit)

            # Публикуем матрицу (32FC1 - это требование задания)
            matrix_msg = self.bridge.cv2_to_imgmsg(temp_matrix, encoding="32FC1")
            self.matrix_pub.publish(matrix_msg)

            # 2. Создаем цветное термо-изображение (IRONBOW)
            # Сначала нормализуем данные для отображения
            norm_frame = cv2.normalize(temp_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # ДЛЯ IRONBOW ИСПОЛЬЗУЕМ COLORMAP_MAGMA
            color_image = cv2.applyColorMap(norm_frame, cv2.COLORMAP_MAGMA)

            # Публикуем основное термо-изображение
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(color_image, "bgr8"))

            # 3. Debug-изображение (как на рис. 4 в задании)
            debug_img = color_image.copy()

            # Находим Min, Max и Центр для подписей
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_matrix)
            center_coords = (self.width // 2, self.height // 2)
            center_val = temp_matrix[center_coords[1], center_coords[0]]

            # Рисуем метки (Max - красный, Min - синий, Центр - желтый)
            self.draw_marker(debug_img, max_loc, max_val, (0, 0, 255), "Max")
            self.draw_marker(debug_img, min_loc, min_val, (255, 0, 0), "Min")
            self.draw_marker(debug_img, center_coords, center_val, (0, 255, 255), "C")

            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

        except Exception as e:
            rospy.logerr(f"Ошибка обработки кадра: {e}")

    def draw_marker(self, img, pos, val, color, label):
        x, y = pos
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 10, 1)
        cv2.putText(img, f"{val:.1f}C", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


if __name__ == '__main__':
    try:
        MileseeyThermalNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
