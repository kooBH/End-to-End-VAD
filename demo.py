import cv2
 
if __name__=='__main__':

    #cap = cv2.VideoCapture(0)   # 0: default camera
    cap = cv2.VideoCapture("./data/Speaker1.avi") #동영상 파일에서 읽기

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 3
     
    while cap.isOpened():
        # 카메라 프레임 읽기
        success, frame = cap.read()

        if success:
            # frame.shape : height , width, channel
            h = frame.shape[0]
            w = frame.shape[1]
            frame = cv2.putText(frame,"AA",(0,h),font,fontScale,(0,255,0),2)

            cv2.imshow('Camera Window', frame)

            # ESC를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if (key == 27): 
                break

    cap.release()
    cv2.destroyAllWindows()


