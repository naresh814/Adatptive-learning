"""
camera_test.py — Quick camera connectivity test
"""
import cv2, time

for idx in [0, 1, 2]:
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        print(f"[OK] Camera found at index {idx}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps_t, fps_f, fps = time.time(), 0, 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            fps_f += 1
            if time.time() - fps_t >= 1.0:
                fps = fps_f; fps_f = 0; fps_t = time.time()
            cv2.putText(frame, f"Camera {idx} | FPS:{fps}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(f"Camera {idx}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        break
    else:
        print(f"[--] No camera at index {idx}")
