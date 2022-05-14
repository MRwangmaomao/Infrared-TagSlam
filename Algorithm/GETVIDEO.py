# -*- coding: UTF-8 –*-
import cv2


def main():
    cap = cv2.VideoCapture(0)

    # 判断摄像头是否打开
    while(cap.isOpened()):
        # 从摄像头获取帧数据
        ret, frame = cap.read()
        # print(frame.shape)
        if ret:
            # 显示帧数据
            cv2.imshow('frame', frame)

            # 如果检测到了按键q则退出，不再显示摄像头并且保存视频文件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 释放摄像头
    cap.release()
    # 删除全部窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()