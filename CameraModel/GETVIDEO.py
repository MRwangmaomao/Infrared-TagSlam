# -*- coding: UTF-8 –*-
import cv2


def balanceImg(img):
    b, g, r = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img


def main():
    cap = cv2.VideoCapture(0)

    # 判断摄像头是否打开
    while(cap.isOpened()):
        # 从摄像头获取帧数据
        ret, frame = cap.read()
        balance_img = balanceImg(frame)
        # print(frame.shape)
        if ret:
            # 显示帧数据
            cv2.imshow('frame', balance_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite(balance_img, "1.jpg")

    # 释放摄像头
    cap.release()
    # 删除全部窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()