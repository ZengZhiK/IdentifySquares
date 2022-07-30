import cv2
import imageio

# 可调参数
win_size_half = 50              # RIO窗口大小的一般，例如设置为50，则RIO窗口大小为100×100
win_center = [330, 240]         # RIO窗口的中心点

in_range_min = 20               # inRange函数的灰度最小值
kernel_size = (7, 7)            # 形态学开运算的kernel大小

radius_threshold_min = 15       # 筛选方块的外接圆半径最小值
radius_threshold_max = 51       # 筛选方块的外接圆半径最大值

identify_count = 0              # ROI窗口检测到疑似方块的次数
identify_count_threshold = 12   # ROI窗口检测到疑似方块的次数 > identify_count_threshold才认为真正检测到方块

color = (0, 0, 255)             # 绘制结果统一采用红色

img_list = []

if __name__ == '__main__':
    # 参考帧为第一帧
    ref_img = cv2.imread('./imgs/0.png', cv2.IMREAD_UNCHANGED)
    img_height, img_width = ref_img.shape

    # 对其余帧进行遍历检测方块
    for i in range(1, 737):
        # ============================================================
        # 读取当前帧
        current_img = cv2.imread('./imgs/{}.png'.format(i), cv2.IMREAD_UNCHANGED)
        # current_img = cv2.imread('./imgs/160.png'.format(i), cv2.IMREAD_UNCHANGED)
        # 与参考帧比较
        frame_diff = cv2.absdiff(ref_img, current_img)

        # ============================================================
        # 设置ROI，并绘制矩形框
        # (x, y)
        top_left = (
            {True: win_center[0] - win_size_half, False: 0}[win_center[0] - win_size_half >= 0],
            {True: win_center[1] - win_size_half, False: 0}[win_center[1] - win_size_half >= 0]
        )
        down_right = (
            {True: win_center[0] + win_size_half, False: img_width - 1}[win_center[0] + win_size_half <= img_width - 1],
            {True: win_center[1] + win_size_half, False: img_height - 1}[
                win_center[1] + win_size_half <= img_height - 1]
        )
        # 从原图中获取H范围，W范围
        frame_diff_roi = frame_diff[top_left[1]:down_right[1], top_left[0]:down_right[0]]

        # ============================================================
        # 二值化
        frame_diff_roi_bin = cv2.inRange(frame_diff_roi, in_range_min, frame_diff_roi.max().item())
        # 形态学运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        frame_diff_roi_bin = cv2.morphologyEx(frame_diff_roi_bin, cv2.MORPH_OPEN, kernel)

        # ============================================================
        # 查找轮廓
        _, contours, _ = cv2.findContours(frame_diff_roi_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
        frame_diff_roi_bin_color = cv2.cvtColor(frame_diff_roi_bin, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(frame_diff_roi_bin_color, contours, -1, color, 2)

        # ============================================================
        # 将当前图片转为彩色图，标注ROI区域和检测出的方块区域
        current_img = cv2.normalize(current_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        current_img_color = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(current_img_color, top_left, down_right, color, 2)
        # 求最小外接圆
        for j in range(0, len(contours)):
            center, radius = cv2.minEnclosingCircle(contours[j])
            # 确认是方块
            if radius_threshold_min <= radius <= radius_threshold_max:
                identify_count += 1
                if identify_count > identify_count_threshold:
                    # 更新ROI窗
                    win_center[0] = (win_center[0] - win_size_half) + int(center[0])
                    win_center[1] = (win_center[1] - win_size_half) + int(center[1])
                    # cv2.circle(frame_diff_roi_bin_color, (int(center[0]), int(center[1])), int(radius), color, 2)
                    cv2.circle(current_img_color, tuple(win_center), int(radius), color, 2)
                    break
            if radius > radius_threshold_max:
                print(center, radius)

        img_list.append(cv2.cvtColor(current_img_color, cv2.COLOR_BGR2RGB))

        # 图像显示
        # ref_img = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        frame_diff = cv2.normalize(frame_diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        frame_diff_roi = cv2.normalize(frame_diff_roi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # frame_diff_roi_bin = cv2.normalize(frame_diff_roi_bin, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # cv2.imshow('ref_img', ref_img)
        cv2.imshow('current_img_color', current_img_color)
        cv2.imshow('frame_diff', frame_diff)
        cv2.imshow('frame_diff_roi', frame_diff_roi)
        # cv2.imshow('frame_diff_roi_bin', frame_diff_roi_bin)
        cv2.imshow('frame_diff_roi_bin_color', frame_diff_roi_bin_color)

        cv2.imwrite('./result/{}.png'.format(i), current_img_color)

        cv2.waitKey(30)
        # cv2.destroyWindow('ref_img')
        # cv2.destroyWindow('current_img')
        # cv2.destroyWindow('frame_diff')
        # cv2.destroyWindow('frame_diff_roi')
        # cv2.destroyWindow('frame_diff_roi_bin')

    imageio.mimsave('./result/video_result.gif', img_list, fps=60)